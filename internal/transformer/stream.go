// Package transformer handles request/response transformation and token counting.
package transformer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"oc-go-cc/pkg/types"
)

// ErrClientDisconnected is returned when the client disconnects during streaming.
var ErrClientDisconnected = fmt.Errorf("client disconnected")

// StreamHandler handles streaming SSE transformation from OpenAI to Anthropic format.
type StreamHandler struct {
	responseTransformer *ResponseTransformer
}

// NewStreamHandler creates a new stream handler.
func NewStreamHandler() *StreamHandler {
	return &StreamHandler{
		responseTransformer: NewResponseTransformer(),
	}
}

// ProxyStream takes an OpenAI streaming response and writes Anthropic-format SSE to the writer.
// It reads OpenAI ChatCompletionChunk SSE events and transforms them into Anthropic MessageEvent SSE events.
// The clientCtx is used to detect client disconnection and abort early.
//
// CRITICAL: This function reads directly from resp.Body without buffering to minimize latency.
// Per deep research: "Don't use bufio.Scanner or bufio.Reader on the response body - it adds buffering"
func (h *StreamHandler) ProxyStream(
	w http.ResponseWriter,
	openaiResp io.ReadCloser,
	originalModel string,
	clientCtx context.Context,
) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("streaming not supported by response writer")
	}

	// Generate a unique message ID for this stream.
	msgID := "msg_" + generateID()

	// Send message_start event with the full message envelope.
	msgStart := types.MessageEvent{
		Type: "message_start",
		Message: &types.MessageResponse{
			ID:      msgID,
			Type:    "message",
			Role:    "assistant",
			Content: []types.ContentBlock{},
			Model:   originalModel,
		},
	}
	if err := writeSSEEvent(w, msgStart); err != nil {
		return ErrClientDisconnected
	}
	flusher.Flush()

	// Read directly from response body without buffering.
	// Use a tight loop with a line buffer - no bufio.Reader.
	var lineBuf bytes.Buffer
	state := &streamState{
		textIndex:     -1,
		thinkingIndex: -1,
		toolBlocks:    make(map[int]int),
		openBlocks:    make(map[int]bool),
	}

	// Read in larger chunks for efficiency, then parse lines
	readBuf := make([]byte, 4096)

	for {
		// Check if client disconnected
		select {
		case <-clientCtx.Done():
			return ErrClientDisconnected
		default:
		}

		// Read chunk from upstream
		n, err := openaiResp.Read(readBuf)
		if n > 0 {
			// Process bytes immediately
			for i := 0; i < n; i++ {
				b := readBuf[i]
				if b == '\n' {
					line := lineBuf.String()
					lineBuf.Reset()

					// Process complete line
					if err := h.processSSELine(w, flusher, line, state, originalModel); err != nil {
						return err
					}
				} else {
					lineBuf.WriteByte(b)
				}
			}
		}

		if err == io.EOF {
			// Process any remaining data in buffer
			if lineBuf.Len() > 0 {
				line := lineBuf.String()
				if err := h.processSSELine(w, flusher, line, state, originalModel); err != nil {
					return err
				}
			}
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read stream: %w", err)
		}
	}

	// Send message_stop event to signal stream completion.
	stopEvent := types.MessageEvent{
		Type: "message_stop",
	}
	if err := writeSSEEvent(w, stopEvent); err != nil {
		return ErrClientDisconnected
	}
	flusher.Flush()

	return nil
}

type streamState struct {
	nextIndex     int
	textIndex     int
	thinkingIndex int
	toolBlocks    map[int]int
	openBlocks    map[int]bool
}

func (s *streamState) ensureThinkingBlock(w http.ResponseWriter) (int, error) {
	if s.thinkingIndex != -1 && s.openBlocks[s.thinkingIndex] {
		return s.thinkingIndex, nil
	}

	idx := s.nextIndex
	s.nextIndex++
	s.thinkingIndex = idx
	s.openBlocks[idx] = true

	startEvent := types.MessageEvent{
		Type:         "content_block_start",
		Index:        &idx,
		ContentBlock: &types.ContentBlock{Type: "thinking", Thinking: ""},
	}
	if err := writeSSEEvent(w, startEvent); err != nil {
		return idx, ErrClientDisconnected
	}
	return idx, nil
}

func (s *streamState) stopThinkingBlock(w http.ResponseWriter) error {
	if s.thinkingIndex == -1 || !s.openBlocks[s.thinkingIndex] {
		return nil
	}
	idx := s.thinkingIndex
	if err := writeSSEEvent(w, types.MessageEvent{Type: "content_block_stop", Index: &idx}); err != nil {
		return ErrClientDisconnected
	}
	delete(s.openBlocks, idx)
	s.thinkingIndex = -1
	return nil
}

func (s *streamState) ensureTextBlock(w http.ResponseWriter) (int, error) {
	if s.textIndex != -1 && s.openBlocks[s.textIndex] {
		return s.textIndex, nil
	}

	idx := s.nextIndex
	s.nextIndex++
	s.textIndex = idx
	s.openBlocks[idx] = true

	startEvent := types.MessageEvent{
		Type:         "content_block_start",
		Index:        &idx,
		ContentBlock: &types.ContentBlock{Type: "text", Text: ""},
	}
	if err := writeSSEEvent(w, startEvent); err != nil {
		return idx, ErrClientDisconnected
	}
	return idx, nil
}

func (s *streamState) stopTextBlock(w http.ResponseWriter) error {
	if s.textIndex == -1 || !s.openBlocks[s.textIndex] {
		return nil
	}
	idx := s.textIndex
	if err := writeSSEEvent(w, types.MessageEvent{Type: "content_block_stop", Index: &idx}); err != nil {
		return ErrClientDisconnected
	}
	delete(s.openBlocks, idx)
	s.textIndex = -1
	return nil
}

func (s *streamState) ensureToolBlock(w http.ResponseWriter, streamIndex int, tc types.ToolCall) (int, error) {
	if idx, ok := s.toolBlocks[streamIndex]; ok {
		return idx, nil
	}

	idx := s.nextIndex
	s.nextIndex++
	s.toolBlocks[streamIndex] = idx
	s.openBlocks[idx] = true

	toolID := tc.ID
	if toolID == "" {
		toolID = fmt.Sprintf("toolu_%s_%d", generateID(), streamIndex)
	}
	toolName := tc.Function.Name
	if toolName == "" {
		toolName = fmt.Sprintf("tool_%d", streamIndex)
	}

	input := json.RawMessage(`{}`)
	startEvent := types.MessageEvent{
		Type:  "content_block_start",
		Index: &idx,
		ContentBlock: &types.ContentBlock{
			Type:  "tool_use",
			ID:    toolID,
			Name:  toolName,
			Input: input,
		},
	}
	if err := writeSSEEvent(w, startEvent); err != nil {
		return idx, ErrClientDisconnected
	}
	return idx, nil
}

func (s *streamState) stopOpenBlocks(w http.ResponseWriter) error {
	if len(s.openBlocks) == 0 {
		return nil
	}

	indices := make([]int, 0, len(s.openBlocks))
	for idx := range s.openBlocks {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	for _, idx := range indices {
		if err := writeSSEEvent(w, types.MessageEvent{Type: "content_block_stop", Index: &idx}); err != nil {
			return ErrClientDisconnected
		}
		delete(s.openBlocks, idx)
	}
	s.textIndex = -1
	s.thinkingIndex = -1
	return nil
}

// processSSELine processes a single SSE line from upstream.
// Per deep research: "Treat SSE primarily as a text protocol" - minimize JSON parsing.
func (h *StreamHandler) processSSELine(
	w http.ResponseWriter,
	flusher http.Flusher,
	line string,
	state *streamState,
	originalModel string,
) error {
	line = strings.TrimSpace(line)

	// Skip empty lines
	if line == "" {
		return nil
	}

	// Skip non-data lines (event: lines, id: lines, etc.)
	if !strings.HasPrefix(line, "data: ") {
		return nil
	}

	data := strings.TrimPrefix(line, "data: ")
	if data == "" {
		return nil
	}

	// Handle [DONE] marker
	if data == "[DONE]" {
		return nil
	}

	// Fast path: check if this is a content chunk without full JSON parsing.
	// Skip the fast path when reasoning_content is present in the same chunk so
	// the JSON path can preserve both reasoning and text deltas.
	if !strings.Contains(data, `"reasoning_content"`) {
		if idx := strings.Index(data, `"delta":{"content":"`); idx != -1 {
			// Extract content directly
			start := idx + len(`"delta":{"content":"`)
			content, ok := extractJSONStringValue(data, start)
			if ok {
				if content != "" {
					if err := state.stopThinkingBlock(w); err != nil {
						return err
					}
					blockIndex, err := state.ensureTextBlock(w)
					if err != nil {
						return err
					}

					// Send content_block_delta
					delta := types.Delta{
						Type: "text_delta",
						Text: content,
					}
					event := types.MessageEvent{
						Type:  "content_block_delta",
						Index: &blockIndex,
						Delta: &delta,
					}
					if err := writeSSEEvent(w, event); err != nil {
						return ErrClientDisconnected
					}
					flusher.Flush()
					return nil
				}
			}
		}
	}

	// Check for finish_reason - need to send stop events. If the chunk also has
	// usage, fall through to full JSON parsing so usage is preserved.
	if strings.Contains(data, `"finish_reason":`) &&
		!strings.Contains(data, `"finish_reason":null`) &&
		!strings.Contains(data, `"usage":`) {
		if err := state.stopOpenBlocks(w); err != nil {
			return err
		}

		stopReason := "end_turn"
		if strings.Contains(data, `"finish_reason":"tool_calls"`) {
			stopReason = "tool_use"
		} else if strings.Contains(data, `"finish_reason":"length"`) {
			stopReason = "max_tokens"
		}

		// Send message_delta with stop_reason
		msgDelta := types.MessageEvent{
			Type: "message_delta",
			Delta: &types.Delta{
				StopReason: stopReason,
			},
		}
		if err := writeSSEEvent(w, msgDelta); err != nil {
			return ErrClientDisconnected
		}
		flusher.Flush()
		return nil
	}

	// For tool calls and other complex cases, fall back to full JSON parsing
	if upstreamErr := parseStreamError(data); upstreamErr != nil {
		return upstreamErr
	}

	var chunk types.ChatCompletionChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		// Skip malformed chunks - don't fail the whole stream
		return nil
	}

	if len(chunk.Choices) == 0 {
		if chunk.Usage != nil {
			if err := h.sendUsageDelta(w, flusher, chunk.Usage); err != nil {
				return err
			}
		}
		return nil
	}

	choice := chunk.Choices[0]

	// Handle DeepSeek/OpenAI-compatible reasoning deltas. Claude Code expects
	// these to round-trip as Anthropic thinking blocks in the conversation
	// history, otherwise DeepSeek rejects the next thinking-mode request.
	if choice.Delta.ReasoningContent != nil && *choice.Delta.ReasoningContent != "" {
		if err := state.stopTextBlock(w); err != nil {
			return err
		}
		blockIndex, err := state.ensureThinkingBlock(w)
		if err != nil {
			return err
		}

		delta := types.Delta{
			Type:     "thinking_delta",
			Thinking: *choice.Delta.ReasoningContent,
		}
		event := types.MessageEvent{
			Type:  "content_block_delta",
			Index: &blockIndex,
			Delta: &delta,
		}
		if err := writeSSEEvent(w, event); err != nil {
			return ErrClientDisconnected
		}
		flusher.Flush()
	}

	// Handle text content deltas
	if choice.Delta.Content != "" {
		if err := state.stopThinkingBlock(w); err != nil {
			return err
		}
		blockIndex, err := state.ensureTextBlock(w)
		if err != nil {
			return err
		}

		delta := types.Delta{
			Type: "text_delta",
			Text: choice.Delta.Content,
		}
		event := types.MessageEvent{
			Type:  "content_block_delta",
			Index: &blockIndex,
			Delta: &delta,
		}
		if err := writeSSEEvent(w, event); err != nil {
			return ErrClientDisconnected
		}
		flusher.Flush()
	}

	// Handle tool call deltas
	if len(choice.Delta.ToolCalls) > 0 {
		if err := state.stopThinkingBlock(w); err != nil {
			return err
		}
		if err := state.stopTextBlock(w); err != nil {
			return err
		}
		for _, tc := range choice.Delta.ToolCalls {
			streamIndex := len(state.toolBlocks)
			if tc.Index != nil {
				streamIndex = *tc.Index
			}
			blockIndex, err := state.ensureToolBlock(w, streamIndex, tc)
			if err != nil {
				return err
			}

			if tc.Function.Arguments != "" {
				delta := types.Delta{
					Type:        "input_json_delta",
					PartialJSON: tc.Function.Arguments,
				}
				event := types.MessageEvent{
					Type:  "content_block_delta",
					Index: &blockIndex,
					Delta: &delta,
				}
				if err := writeSSEEvent(w, event); err != nil {
					return ErrClientDisconnected
				}
			}
			flusher.Flush()
		}
	}

	// Handle finish reason
	if choice.FinishReason != "" {
		if err := state.stopOpenBlocks(w); err != nil {
			return err
		}

		msgDelta := types.MessageEvent{
			Type: "message_delta",
			Delta: &types.Delta{
				StopReason: h.responseTransformer.mapFinishReason(choice.FinishReason),
			},
			Usage: usageInfoToAnthropic(chunk.Usage),
		}
		if err := writeSSEEvent(w, msgDelta); err != nil {
			return ErrClientDisconnected
		}
		flusher.Flush()
	}

	return nil
}

func parseStreamError(data string) error {
	var envelope struct {
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(data), &envelope); err != nil || envelope.Error == nil {
		return nil
	}

	message := envelope.Error.Message
	if message == "" {
		message = "upstream stream error"
	}
	if envelope.Error.Code != "" {
		return fmt.Errorf("upstream stream error: %s: %s", envelope.Error.Code, message)
	}
	if envelope.Error.Type != "" {
		return fmt.Errorf("upstream stream error: %s: %s", envelope.Error.Type, message)
	}
	return fmt.Errorf("upstream stream error: %s", message)
}

func (h *StreamHandler) sendUsageDelta(w http.ResponseWriter, flusher http.Flusher, usage *types.UsageInfo) error {
	event := types.MessageEvent{
		Type:  "message_delta",
		Usage: usageInfoToAnthropic(usage),
	}
	if err := writeSSEEvent(w, event); err != nil {
		return ErrClientDisconnected
	}
	flusher.Flush()
	return nil
}

func usageInfoToAnthropic(usage *types.UsageInfo) *types.Usage {
	if usage == nil {
		return nil
	}
	return &types.Usage{
		InputTokens:              usage.PromptTokens,
		OutputTokens:             usage.CompletionTokens,
		CacheCreationInputTokens: usage.PromptCacheMissTokens,
		CacheReadInputTokens:     usage.PromptCacheHitTokens,
	}
}

func extractJSONStringValue(data string, start int) (string, bool) {
	escaped := false
	for i := start; i < len(data); i++ {
		switch data[i] {
		case '\\':
			escaped = !escaped
		case '"':
			if escaped {
				escaped = false
				continue
			}
			value, err := strconv.Unquote(`"` + data[start:i] + `"`)
			if err != nil {
				return "", false
			}
			return value, true
		default:
			escaped = false
		}
	}
	return "", false
}

// writeSSEEvent writes a single SSE event to the HTTP response writer.
// Format: "event: <type>\ndata: <json>\n\n"
func writeSSEEvent(w http.ResponseWriter, event types.MessageEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.Type, string(data))
	return err
}

// generateID creates a unique identifier based on current time.
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}
