package handlers

import (
	"encoding/json"
	"testing"
)

func TestReplaceModelInRawBodyOnlyChangesTopLevelModel(t *testing.T) {
	raw := json.RawMessage(`{
		"model": "claude-requested",
		"messages": [{
			"role": "user",
			"content": [{"type": "text", "text": "hello"}]
		}],
		"tool_choice": {
			"type": "tool",
			"name": "pick_model",
			"input": {"model": "nested-should-stay"}
		}
	}`)

	updated := replaceModelInRawBody(raw, "minimax-m2.7")

	var body map[string]interface{}
	if err := json.Unmarshal(updated, &body); err != nil {
		t.Fatalf("updated body is invalid JSON: %v", err)
	}
	if got, want := body["model"], "minimax-m2.7"; got != want {
		t.Fatalf("top-level model = %q, want %q", got, want)
	}

	toolChoice := body["tool_choice"].(map[string]interface{})
	input := toolChoice["input"].(map[string]interface{})
	if got, want := input["model"], "nested-should-stay"; got != want {
		t.Fatalf("nested model = %q, want %q", got, want)
	}
}
