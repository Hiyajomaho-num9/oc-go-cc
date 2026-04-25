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
	state := &streamState{
		textIndex:  -1,
		toolBlocks: make(map[int]int),
		openBlocks: make(map[int]bool),
	}

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
