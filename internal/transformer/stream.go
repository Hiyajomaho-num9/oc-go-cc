// Package transformer handles request/response transformation and token counting.
package transformer

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"oc-go-cc/pkg/types"
)

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
func (h *StreamHandler) ProxyStream(
	w http.ResponseWriter,
	openaiResp io.ReadCloser,
	originalModel string,
) error {
	// Set headers for SSE streaming.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

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
		return fmt.Errorf("failed to write message_start event: %w", err)
	}
	flusher.Flush()

	// Use bufio.Reader for real-time reading (Scanner buffers entire lines).
	reader := bufio.NewReader(openaiResp)
	contentIndex := 0
	var lineBuf bytes.Buffer

	for {
		b, err := reader.ReadByte()
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to read stream: %w", err)
		}

		if b == '\n' {
			line := lineBuf.String()
			lineBuf.Reset()

			// Skip empty lines and non-data lines
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}
			if data == "" {
				continue
			}

			var chunk types.ChatCompletionChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			// Handle text content deltas.
			if choice.Delta.Content != "" {
				// Send content_block_start for the first text delta.
				if contentIndex == 0 {
					startEvent := types.MessageEvent{
						Type:  "content_block_start",
						Index: &contentIndex,
						Delta: &types.Delta{
							Type: "text",
						},
					}
					if err := writeSSEEvent(w, startEvent); err != nil {
						return fmt.Errorf("failed to write content_block_start: %w", err)
					}
				}

				delta := types.Delta{
					Type: "text_delta",
					Text: choice.Delta.Content,
				}

				event := types.MessageEvent{
					Type:  "content_block_delta",
					Index: &contentIndex,
					Delta: &delta,
				}

				if err := writeSSEEvent(w, event); err != nil {
					return fmt.Errorf("failed to write content_block_delta: %w", err)
				}
				flusher.Flush()
			}

			// Handle tool call deltas.
			if len(choice.Delta.ToolCalls) > 0 {
				for _, tc := range choice.Delta.ToolCalls {
					// Increment index for each tool call block.
					contentIndex++

					// Send content_block_start for tool use.
					startEvent := types.MessageEvent{
						Type:  "content_block_start",
						Index: &contentIndex,
						Delta: &types.Delta{
							Type: "tool_use",
						},
					}
					if err := writeSSEEvent(w, startEvent); err != nil {
						return fmt.Errorf("failed to write tool content_block_start: %w", err)
					}

					// Send input_json_delta for tool arguments.
					if tc.Function.Arguments != "" {
						delta := types.Delta{
							Type:        "input_json_delta",
							PartialJSON: tc.Function.Arguments,
						}

						event := types.MessageEvent{
							Type:  "content_block_delta",
							Index: &contentIndex,
							Delta: &delta,
						}

						if err := writeSSEEvent(w, event); err != nil {
							return fmt.Errorf("failed to write input_json_delta: %w", err)
						}
					}
					flusher.Flush()
				}
			}

			// Handle finish reason — stream is ending.
			if choice.FinishReason != "" {
				// Send content_block_stop for the last active block.
				stopEvent := types.MessageEvent{
					Type:  "content_block_stop",
					Index: &contentIndex,
				}
				if err := writeSSEEvent(w, stopEvent); err != nil {
					return fmt.Errorf("failed to write content_block_stop: %w", err)
				}

				// Build usage delta from chunk usage if available.
				var usage *types.Usage
				if chunk.Usage != nil {
					usage = &types.Usage{
						InputTokens:  chunk.Usage.PromptTokens,
						OutputTokens: chunk.Usage.CompletionTokens,
					}
				}

				// Send message_delta with stop reason and usage.
				msgDelta := types.MessageEvent{
					Type: "message_delta",
					Delta: &types.Delta{
						StopReason: h.responseTransformer.mapFinishReason(choice.FinishReason),
					},
					Usage: usage,
				}
				if err := writeSSEEvent(w, msgDelta); err != nil {
					return fmt.Errorf("failed to write message_delta: %w", err)
				}

				flusher.Flush()
			}
		} else {
			lineBuf.WriteByte(b)
		}
	}

	// Send message_stop event to signal stream completion.
	stopEvent := types.MessageEvent{
		Type: "message_stop",
	}
	if err := writeSSEEvent(w, stopEvent); err != nil {
		return fmt.Errorf("failed to write message_stop: %w", err)
	}
	flusher.Flush()

	return nil
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
