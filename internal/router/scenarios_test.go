package router

import (
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
