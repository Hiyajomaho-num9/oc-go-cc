// Package metrics provides in-memory metrics for the proxy.
package metrics

import (
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Metrics holds in-memory metrics for the proxy.
type Metrics struct {
	// Counters (atomic)
	requestsReceived atomic.Int64
	requestsStreamed atomic.Int64
	requestsSuccess  atomic.Int64
	requestsFailed   atomic.Int64
	upstreamCalls    atomic.Int64
	rateLimited      atomic.Int64
	deduplicated     atomic.Int64

	// Latency tracking
	mu                sync.RWMutex
	latencies         []time.Duration
	maxLatencySamples int

	// By model
	modelCounts map[string]*atomic.Int64
	modelMu     sync.RWMutex
}

// New creates a new metrics instance.
func New() *Metrics {
	return &Metrics{
		maxLatencySamples: 1000,
		modelCounts:       make(map[string]*atomic.Int64),
	}
}

// RecordRequest records an incoming request.
func (m *Metrics) RecordRequest(streaming bool) {
	m.requestsReceived.Add(1)
	if streaming {
		m.requestsStreamed.Add(1)
	}
}

// RecordSuccess records a successful request.
func (m *Metrics) RecordSuccess(model string, latency time.Duration) {
	m.requestsSuccess.Add(1)
	m.upstreamCalls.Add(1)
	m.recordLatency(latency)
	m.recordModel(model)
}

// RecordFailure records a failed request.
func (m *Metrics) RecordFailure() {
	m.requestsFailed.Add(1)
}

// RecordRateLimited records a rate-limited request.
func (m *Metrics) RecordRateLimited() {
	m.rateLimited.Add(1)
}

// RecordDeduplicated records a deduplicated request.
func (m *Metrics) RecordDeduplicated() {
	m.deduplicated.Add(1)
}

func (m *Metrics) recordLatency(latency time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Keep last N samples for p95/p99
	if len(m.latencies) >= m.maxLatencySamples {
		// Shift and add new
		m.latencies = m.latencies[1:]
	}
	m.latencies = append(m.latencies, latency)
}

func (m *Metrics) recordModel(model string) {
	m.modelMu.Lock()
	defer m.modelMu.Unlock()

	if _, exists := m.modelCounts[model]; !exists {
		m.modelCounts[model] = &atomic.Int64{}
	}
	m.modelCounts[model].Add(1)
}

// GetSnapshot returns a snapshot of current metrics.
func (m *Metrics) GetSnapshot() Snapshot {
	m.mu.RLock()
	latencies := make([]time.Duration, len(m.latencies))
	copy(latencies, m.latencies)
	m.mu.RUnlock()

	modelCounts := make(map[string]int64)
	m.modelMu.RLock()
	for k, v := range m.modelCounts {
		modelCounts[k] = v.Load()
	}
	m.modelMu.RUnlock()

	return Snapshot{
		RequestsReceived: m.requestsReceived.Load(),
		RequestsStreamed: m.requestsStreamed.Load(),
		RequestsSuccess:  m.requestsSuccess.Load(),
		RequestsFailed:   m.requestsFailed.Load(),
		UpstreamCalls:    m.upstreamCalls.Load(),
		RateLimited:      m.rateLimited.Load(),
		Deduplicated:     m.deduplicated.Load(),
		Latencies:        latencies,
		ModelCounts:      modelCounts,
	}
}

// Snapshot represents a point-in-time view of metrics.
type Snapshot struct {
	RequestsReceived int64
	RequestsStreamed int64
	RequestsSuccess  int64
	RequestsFailed   int64
	UpstreamCalls    int64
	RateLimited      int64
	Deduplicated     int64
	Latencies        []time.Duration
	ModelCounts      map[string]int64
}

// CalculateP95 calculates the p95 latency from the snapshot.
func (s Snapshot) CalculateP95() time.Duration {
	if len(s.Latencies) == 0 {
		return 0
	}
	return percentile(s.Latencies, 0.95)
}

// CalculateP99 calculates the p99 latency from the snapshot.
func (s Snapshot) CalculateP99() time.Duration {
	if len(s.Latencies) == 0 {
		return 0
	}
	return percentile(s.Latencies, 0.99)
}

func percentile(latencies []time.Duration, p float64) time.Duration {
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i] < sorted[j]
	})

	index := int(float64(len(sorted)) * p)
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}
