package middleware

import (
	"encoding/json"
	"testing"
	"time"
)

func TestRequestDeduplicatorRejectsDuplicateUntilRelease(t *testing.T) {
	dedup := NewRequestDeduplicator(time.Second)
	body := json.RawMessage(`{"model":"deepseek-v4-pro","messages":[]}`)

	if _, ok := dedup.TryAcquire(body); !ok {
		t.Fatal("first TryAcquire() returned false, want true")
	}
	if _, ok := dedup.TryAcquire(body); ok {
		t.Fatal("second TryAcquire() returned true, want false")
	}

	dedup.Release(body)
	if _, ok := dedup.TryAcquire(body); !ok {
		t.Fatal("TryAcquire() after Release returned false, want true")
	}
}
