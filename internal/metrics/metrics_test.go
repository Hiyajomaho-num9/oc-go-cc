package metrics

import (
	"testing"
	"time"
)

func TestSnapshotPercentilesSortLatencies(t *testing.T) {
	snapshot := Snapshot{
		Latencies: []time.Duration{
			400 * time.Millisecond,
			100 * time.Millisecond,
			300 * time.Millisecond,
			200 * time.Millisecond,
		},
	}

	if got, want := snapshot.CalculateP95(), 400*time.Millisecond; got != want {
		t.Fatalf("CalculateP95() = %s, want %s", got, want)
	}
	if got, want := snapshot.CalculateP99(), 400*time.Millisecond; got != want {
		t.Fatalf("CalculateP99() = %s, want %s", got, want)
	}
}
