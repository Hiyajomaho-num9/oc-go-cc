package transformer

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"oc-go-cc/pkg/types"
)

func TestProcessSSELineFastPathUnescapesJSONContent(t *testing.T) {
	handler := NewStreamHandler()
	recorder := httptest.NewRecorder()
	state := newTestStreamState()

	line := `data: {"id":"chunk_1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello\nworld \"quoted\" \\slash"},"finish_reason":null}]}`
	if err := handler.processSSELine(recorder, recorder, line, state, "deepseek-v4-pro"); err != nil {
		t.Fatalf("processSSELine returned error: %v", err)
	}

	events := decodeSSEEvents(t, recorder.Body.String())
	if len(events) != 2 {
		t.Fatalf("expected content_block_start and content_block_delta, got %d events: %s", len(events), recorder.Body.String())
	}
	if events[1].Delta == nil {
		t.Fatalf("expected delta event, got %#v", events[1])
	}

	want := "hello\nworld \"quoted\" \\slash"
	if got := events[1].Delta.Text; got != want {
		t.Fatalf("delta text mismatch:\n got: %q\nwant: %q", got, want)
	}
}

func TestProcessSSELinePreservesReasoningContentAsThinking(t *testing.T) {
	handler := NewStreamHandler()
	recorder := httptest.NewRecorder()
	state := newTestStreamState()

	reasoningLine := `data: {"id":"chunk_1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"think\nfirst"},"finish_reason":null}]}`
	if err := handler.processSSELine(recorder, recorder, reasoningLine, state, "deepseek-v4-pro"); err != nil {
		t.Fatalf("processSSELine(reasoning) returned error: %v", err)
	}

	textLine := `data: {"id":"chunk_2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}`
	if err := handler.processSSELine(recorder, recorder, textLine, state, "deepseek-v4-pro"); err != nil {
		t.Fatalf("processSSELine(text) returned error: %v", err)
	}

	events := decodeSSEEvents(t, recorder.Body.String())
	if len(events) != 5 {
		t.Fatalf("expected thinking start/delta/stop and text start/delta, got %d events: %s", len(events), recorder.Body.String())
	}

	if got, want := events[0].Type, "content_block_start"; got != want {
		t.Fatalf("events[0].Type = %q, want %q", got, want)
	}
	if events[0].ContentBlock == nil || events[0].ContentBlock.Type != "thinking" {
		t.Fatalf("events[0].ContentBlock = %#v, want thinking block", events[0].ContentBlock)
	}
	if events[1].Delta == nil || events[1].Delta.Type != "thinking_delta" {
		t.Fatalf("events[1].Delta = %#v, want thinking_delta", events[1].Delta)
	}
	if got, want := events[1].Delta.Thinking, "think\nfirst"; got != want {
		t.Fatalf("thinking delta mismatch:\n got: %q\nwant: %q", got, want)
	}
	if got, want := events[2].Type, "content_block_stop"; got != want {
		t.Fatalf("events[2].Type = %q, want %q", got, want)
	}
	if events[3].ContentBlock == nil || events[3].ContentBlock.Type != "text" {
		t.Fatalf("events[3].ContentBlock = %#v, want text block", events[3].ContentBlock)
	}
	if events[4].Delta == nil || events[4].Delta.Text != "answer" {
		t.Fatalf("events[4].Delta = %#v, want text answer", events[4].Delta)
	}
}

func TestProcessSSELineHandlesReasoningAndContentInSameChunk(t *testing.T) {
	handler := NewStreamHandler()
	recorder := httptest.NewRecorder()
	state := newTestStreamState()

	line := `data: {"id":"chunk_1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"think","content":"answer"},"finish_reason":null}]}`
	if err := handler.processSSELine(recorder, recorder, line, state, "deepseek-v4-pro"); err != nil {
		t.Fatalf("processSSELine returned error: %v", err)
	}

	events := decodeSSEEvents(t, recorder.Body.String())
	if len(events) != 5 {
		t.Fatalf("expected thinking start/delta/stop and text start/delta, got %d events: %s", len(events), recorder.Body.String())
	}
	if events[0].ContentBlock == nil || events[0].ContentBlock.Type != "thinking" {
		t.Fatalf("events[0].ContentBlock = %#v, want thinking block", events[0].ContentBlock)
	}
	if events[1].Delta == nil || events[1].Delta.Thinking != "think" {
		t.Fatalf("events[1].Delta = %#v, want thinking delta", events[1].Delta)
	}
	if got, want := events[2].Type, "content_block_stop"; got != want {
		t.Fatalf("events[2].Type = %q, want %q", got, want)
	}
	if events[3].ContentBlock == nil || events[3].ContentBlock.Type != "text" {
		t.Fatalf("events[3].ContentBlock = %#v, want text block", events[3].ContentBlock)
	}
	if events[4].Delta == nil || events[4].Delta.Text != "answer" {
		t.Fatalf("events[4].Delta = %#v, want text answer", events[4].Delta)
	}
}

func TestProcessSSELineEmitsUsageOnlyChunk(t *testing.T) {
	handler := NewStreamHandler()
	recorder := httptest.NewRecorder()
	state := newTestStreamState()

	line := `data: {"id":"chunk_usage","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":123,"completion_tokens":45,"total_tokens":168,"prompt_cache_hit_tokens":100,"prompt_cache_miss_tokens":23}}`
	if err := handler.processSSELine(recorder, recorder, line, state, "deepseek-v4-pro"); err != nil {
		t.Fatalf("processSSELine returned error: %v", err)
	}

	events := decodeSSEEvents(t, recorder.Body.String())
	if len(events) != 1 {
		t.Fatalf("expected one usage event, got %d events: %s", len(events), recorder.Body.String())
	}
	if got, want := events[0].Type, "message_delta"; got != want {
		t.Fatalf("events[0].Type = %q, want %q", got, want)
	}
	if events[0].Usage == nil {
		t.Fatal("events[0].Usage = nil, want usage")
	}
	if got, want := events[0].Usage.InputTokens, 123; got != want {
		t.Fatalf("InputTokens = %d, want %d", got, want)
	}
	if got, want := events[0].Usage.OutputTokens, 45; got != want {
		t.Fatalf("OutputTokens = %d, want %d", got, want)
	}
	if got, want := events[0].Usage.CacheReadInputTokens, 100; got != want {
		t.Fatalf("CacheReadInputTokens = %d, want %d", got, want)
	}
	if got, want := events[0].Usage.CacheCreationInputTokens, 23; got != want {
		t.Fatalf("CacheCreationInputTokens = %d, want %d", got, want)
	}
}

func TestProcessSSELineReturnsUpstreamError(t *testing.T) {
	handler := NewStreamHandler()
	recorder := httptest.NewRecorder()
	state := newTestStreamState()

	line := `data: {"error":{"message":"stream disconnected before completion","type":"server_error","code":"internal_server_error"}}`
	err := handler.processSSELine(recorder, recorder, line, state, "deepseek-v4-pro")
	if err == nil {
		t.Fatal("processSSELine returned nil, want upstream error")
	}
	if !strings.Contains(err.Error(), "internal_server_error") {
		t.Fatalf("error = %q, want upstream error code", err.Error())
	}
}

func newTestStreamState() *streamState {
	return &streamState{
		textIndex:     -1,
		thinkingIndex: -1,
		toolBlocks:    make(map[int]int),
		openBlocks:    make(map[int]bool),
	}
}

func decodeSSEEvents(t *testing.T, body string) []types.MessageEvent {
	t.Helper()

	var events []types.MessageEvent
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		var event types.MessageEvent
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			t.Fatalf("failed to unmarshal SSE event %q: %v", line, err)
		}
		events = append(events, event)
	}
	return events
}
