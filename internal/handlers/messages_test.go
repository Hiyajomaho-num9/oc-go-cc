package handlers

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"oc-go-cc/internal/config"
	"oc-go-cc/internal/metrics"
	"oc-go-cc/internal/middleware"
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

func TestHandleMessagesDuplicateReturnsConflict(t *testing.T) {
	body := []byte(`{"model":"deepseek-v4-pro","max_tokens":1,"messages":[{"role":"user","content":"hello"}]}`)
	handler := &MessagesHandler{
		logger:       slog.Default(),
		rateLimiter:  middleware.NewRateLimiter(100, time.Minute),
		requestDedup: middleware.NewRequestDeduplicator(time.Second),
		requestIDGen: middleware.NewRequestIDGenerator(),
		metrics:      metrics.New(),
	}
	if _, ok := handler.requestDedup.TryAcquire(json.RawMessage(body)); !ok {
		t.Fatal("failed to pre-acquire request dedup slot")
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	handler.HandleMessages(recorder, req)

	if got, want := recorder.Code, http.StatusConflict; got != want {
		t.Fatalf("status = %d, want %d; body: %s", got, want, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "duplicate request skipped") {
		t.Fatalf("body = %q, want duplicate error", recorder.Body.String())
	}
}

func TestStreamingAttemptTimeoutUsesStreamTimeout(t *testing.T) {
	handler := &MessagesHandler{
		config: &config.Config{
			OpenCodeGo: config.OpenCodeGoConfig{
				TimeoutMs:       300000,
				StreamTimeoutMs: 600000,
			},
		},
	}

	if got, want := handler.streamingAttemptTimeout(), 10*time.Minute; got != want {
		t.Fatalf("streamingAttemptTimeout() = %s, want %s", got, want)
	}
}

func TestStreamingAttemptTimeoutFallsBackToRequestTimeout(t *testing.T) {
	handler := &MessagesHandler{
		config: &config.Config{
			OpenCodeGo: config.OpenCodeGoConfig{
				TimeoutMs: 420000,
			},
		},
	}

	if got, want := handler.streamingAttemptTimeout(), 7*time.Minute; got != want {
		t.Fatalf("streamingAttemptTimeout() = %s, want %s", got, want)
	}
}

func TestRedactJSONForLogMasksSecretsAndTruncatesPromptText(t *testing.T) {
	body := []byte(`{
		"api_key": "sk-secret",
		"messages": [{
			"role": "user",
			"content": "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
		}]
	}`)

	redacted := redactJSONForLog(body)
	if strings.Contains(redacted, "sk-secret") {
		t.Fatalf("redacted log still contains secret: %s", redacted)
	}
	if !strings.Contains(redacted, "[redacted]") {
		t.Fatalf("redacted log = %s, want redacted marker", redacted)
	}
	if !strings.Contains(redacted, "truncated") {
		t.Fatalf("redacted log = %s, want truncated prompt marker", redacted)
	}
}

func TestDedupFailsafeWindowUsesStreamTimeoutPlusBuffer(t *testing.T) {
	cfg := &config.Config{OpenCodeGo: config.OpenCodeGoConfig{StreamTimeoutMs: 600000}}
	if got, want := dedupFailsafeWindow(cfg), 11*time.Minute; got != want {
		t.Fatalf("dedupFailsafeWindow() = %s, want %s", got, want)
	}
}
