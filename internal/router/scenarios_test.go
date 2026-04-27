package router

import (
	"strings"
	"testing"

	"oc-go-cc/internal/config"
)

func TestHasComplexPattern_UserMessage(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "Please refactor this code to use interfaces"},
	}
	if !hasComplexPattern(messages) {
		t.Error("Expected hasComplexPattern to detect 'refactor' in user message")
	}
}

func TestHasComplexPattern_SystemMessage(t *testing.T) {
	messages := []MessageContent{
		{Role: "system", Content: "Please architect the new service"},
	}
	if !hasComplexPattern(messages) {
		t.Error("Expected hasComplexPattern to detect 'architect' in system message")
	}
}

func TestHasComplexPattern_NoMatch(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "Hello, how are you?"},
	}
	if hasComplexPattern(messages) {
		t.Error("Expected hasComplexPattern to not match simple greeting")
	}
}

func TestHasThinkingPattern_UserMessage(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "Think through this problem step by step"},
	}
	if !hasThinkingPattern(messages) {
		t.Error("Expected hasThinkingPattern to detect 'think' and 'step by step' in user message")
	}
}

func TestHasThinkingPattern_SystemMessage(t *testing.T) {
	messages := []MessageContent{
		{Role: "system", Content: "You are a reasoning agent"},
	}
	if !hasThinkingPattern(messages) {
		t.Error("Expected hasThinkingPattern to detect 'reasoning' in system message")
	}
}

func TestHasThinkingPattern_AnthropicThinkingBlock(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "Solve this problem antThinking(thinking block)"},
	}
	if !hasThinkingPattern(messages) {
		t.Error("Expected hasThinkingPattern to detect 'antThinking' block")
	}
}

// mockConfig returns a minimal config for testing
func mockConfig() *config.Config {
	return &config.Config{
		Models: map[string]config.ModelConfig{
			"long_context": {
				ContextThreshold: 60000,
			},
		},
	}
}

func TestDetectScenario_ComplexFromUser(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "Architect a new microservice for user authentication"},
	}
	result := DetectScenario(messages, 100, mockConfig())
	if result.Scenario != ScenarioComplex {
		t.Errorf("Expected ScenarioComplex, got %s", result.Scenario)
	}
}

func TestDetectScenario_ThinkFromUser(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "Analyze the tradeoffs of this design"},
	}
	result := DetectScenario(messages, 100, mockConfig())
	if result.Scenario != ScenarioThink {
		t.Errorf("Expected ScenarioThink, got %s", result.Scenario)
	}
}

func TestDetectScenario_DefaultFromSimpleUserMessage(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "Hello, how are you?"},
	}
	result := DetectScenario(messages, 100, mockConfig())
	if result.Scenario != ScenarioDefault {
		t.Errorf("Expected ScenarioDefault, got %s", result.Scenario)
	}
}

func TestDetectScenario_LongContextTakesPriority(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "Refactor this code"},
	}
	// Token count > 60000 should trigger long_context regardless of content
	result := DetectScenario(messages, 70000, mockConfig())
	if result.Scenario != ScenarioLongContext {
		t.Errorf("Expected ScenarioLongContext, got %s", result.Scenario)
	}
}

func TestDetectScenarioReasonUsesConfiguredModelName(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"complex": {ModelID: "deepseek-v4-pro"},
			"default": {ModelID: "deepseek-v4-flash"},
		},
	}

	result := DetectScenario([]MessageContent{{Role: "user", Content: "Refactor this service"}}, 100, cfg)
	if !strings.Contains(result.Reason, "deepseek-v4-pro") {
		t.Fatalf("complex reason = %q, want configured model name", result.Reason)
	}

	result = DetectScenario([]MessageContent{{Role: "user", Content: "hello"}}, 100, cfg)
	if !strings.Contains(result.Reason, "deepseek-v4-flash") {
		t.Fatalf("default reason = %q, want configured model name", result.Reason)
	}
}

func TestRouteForStreamingUsesConfiguredLongContextThreshold(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"long_context": {
				ModelID:          "deepseek-v4-pro",
				ContextThreshold: 1000000,
			},
		},
	}
	messages := []MessageContent{
		{Role: "user", Content: "hello"},
	}

	result := RouteForStreaming(messages, 35000, cfg)
	if result.Scenario == ScenarioLongContext {
		t.Fatalf("RouteForStreaming() = %s, want non-long_context below configured threshold", result.Scenario)
	}

	result = RouteForStreaming(messages, 1000001, cfg)
	if result.Scenario != ScenarioLongContext {
		t.Fatalf("RouteForStreaming() = %s, want %s above configured threshold", result.Scenario, ScenarioLongContext)
	}
	if !strings.Contains(result.Reason, "deepseek-v4-pro") {
		t.Fatalf("RouteForStreaming() reason = %q, want configured model name", result.Reason)
	}
}

func TestRouteForStreamingUsesDefaultLongContextThreshold(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "hello"},
	}

	result := RouteForStreaming(messages, 90000, &config.Config{Models: map[string]config.ModelConfig{}})
	if result.Scenario == ScenarioLongContext {
		t.Fatalf("RouteForStreaming() = %s, want non-long_context below default threshold", result.Scenario)
	}

	result = RouteForStreaming(messages, 100001, &config.Config{Models: map[string]config.ModelConfig{}})
	if result.Scenario != ScenarioLongContext {
		t.Fatalf("RouteForStreaming() = %s, want %s above default threshold", result.Scenario, ScenarioLongContext)
	}
}

func TestRouteForStreamingHandlesNilConfig(t *testing.T) {
	messages := []MessageContent{
		{Role: "user", Content: "hello"},
	}

	result := RouteForStreaming(messages, 90000, nil)
	if result.Scenario == ScenarioLongContext {
		t.Fatalf("RouteForStreaming() = %s, want non-long_context below default threshold with nil config", result.Scenario)
	}

	result = RouteForStreaming(messages, 100001, nil)
	if result.Scenario != ScenarioLongContext {
		t.Fatalf("RouteForStreaming() = %s, want %s above default threshold with nil config", result.Scenario, ScenarioLongContext)
	}
}

func TestGetModelChainDeduplicatesByModelID(t *testing.T) {
	route := RouteResult{
		Primary: config.ModelConfig{ModelID: "deepseek-v4-pro"},
		Fallbacks: []config.ModelConfig{
			{ModelID: "deepseek-v4-pro"},
			{ModelID: "deepseek-v4-flash"},
			{ModelID: "deepseek-v4-flash"},
			{ModelID: "kimi-k2.6"},
		},
	}

	chain := route.GetModelChain()
	if got, want := len(chain), 3; got != want {
		t.Fatalf("len(chain) = %d, want %d", got, want)
	}

	want := []string{"deepseek-v4-pro", "deepseek-v4-flash", "kimi-k2.6"}
	for i, modelID := range want {
		if got := chain[i].ModelID; got != modelID {
			t.Fatalf("chain[%d].ModelID = %q, want %q", i, got, modelID)
		}
	}
}
