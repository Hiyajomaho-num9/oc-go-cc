package router

import (
	"fmt"
	"strings"

	"oc-go-cc/internal/config"
)

// Scenario represents the routing scenario for model selection.
type Scenario string

const (
	ScenarioDefault     Scenario = "default"
	ScenarioBackground  Scenario = "background"
	ScenarioThink       Scenario = "think"
	ScenarioComplex     Scenario = "complex"
	ScenarioLongContext Scenario = "long_context"
	ScenarioFast        Scenario = "fast"
)

// ScenarioResult contains the detected scenario and token count.
type ScenarioResult struct {
	Scenario   Scenario
	TokenCount int
	Reason     string
}

// MessageContent represents a single message in a conversation.
type MessageContent struct {
	Role    string
	Content string
}

// DetectScenario analyzes a request to determine which model to use.
// Routing priority:
//  1. Long Context (> threshold)
//  2. Complex (architectural patterns or tool-heavy operations)
//  3. Think (reasoning patterns)
//  4. Background (simple operations with NO tools)
//  5. Default
//
// For streaming requests, consider using RouteForStreaming() to prefer faster models.
func DetectScenario(messages []MessageContent, tokenCount int, cfg *config.Config) ScenarioResult {
	// 1. Check for long context first (most important)
	threshold := getLongContextThreshold(cfg)
	if tokenCount > threshold {
		model := scenarioModelName(cfg, ScenarioLongContext)
		return ScenarioResult{
			Scenario:   ScenarioLongContext,
			TokenCount: tokenCount,
			Reason:     fmt.Sprintf("token count %d exceeds threshold %d (use %s for long context)", tokenCount, threshold, model),
		}
	}

	// 2. Check for complex tasks (architectural OR tool-related)
	if hasComplexPattern(messages) {
		return ScenarioResult{
			Scenario:   ScenarioComplex,
			TokenCount: tokenCount,
			Reason:     fmt.Sprintf("complex or tool-based operation detected (use %s)", scenarioModelName(cfg, ScenarioComplex)),
		}
	}

	// 3. Check for thinking/reasoning patterns
	if hasThinkingPattern(messages) {
		return ScenarioResult{
			Scenario:   ScenarioThink,
			TokenCount: tokenCount,
			Reason:     fmt.Sprintf("thinking/reasoning pattern detected (use %s)", scenarioModelName(cfg, ScenarioThink)),
		}
	}

	// 4. Check for background task patterns (truly simple operations)
	if hasBackgroundPattern(messages) {
		return ScenarioResult{
			Scenario:   ScenarioBackground,
			TokenCount: tokenCount,
			Reason:     fmt.Sprintf("simple background task detected (use %s)", scenarioModelName(cfg, ScenarioBackground)),
		}
	}

	// 5. Default
	return ScenarioResult{
		Scenario:   ScenarioDefault,
		TokenCount: tokenCount,
		Reason:     fmt.Sprintf("default scenario (use %s)", scenarioModelName(cfg, ScenarioDefault)),
	}
}

// hasComplexPattern looks for complex operations that need more capable models.
// This includes tool-based operations (executing functions, writing/editing files, etc.)
func hasComplexPattern(messages []MessageContent) bool {
	complexKeywords := []string{
		// Architectural
		"architect", "architecture", "refactor", "redesign",
		"complex", "difficult", "challenging",
		"optimize", "performance", "efficiency",
		"design pattern", "best practice",
		// Tool-related keywords indicate complex operations
		"execute", "run command", "bash", "shell",
		"implement", "build", "create", "add feature",
		"write to", "edit file", "create file",
	}

	for _, msg := range messages {
		if msg.Role == "system" || msg.Role == "user" {
			lower := strings.ToLower(msg.Content)
			for _, kw := range complexKeywords {
				if strings.Contains(lower, kw) {
					return true
				}
			}
		}
	}
	return false
}

// hasThinkingPattern looks for system prompts mentioning reasoning keywords
// or content containing thinking/reasoning markers.
func hasThinkingPattern(messages []MessageContent) bool {
	thinkingKeywords := []string{
		"think", "thinking", "plan", "reason", "reasoning",
		"analyze", "analysis", "step by step",
	}

	for _, msg := range messages {
		if msg.Role == "system" || msg.Role == "user" {
			lower := strings.ToLower(msg.Content)
			for _, kw := range thinkingKeywords {
				if strings.Contains(lower, kw) {
					return true
				}
			}
		}
		// Check for thinking content blocks
		if strings.Contains(msg.Content, "antThinking") {
			return true
		}
	}
	return false
}

// hasBackgroundPattern checks for VERY simple background tasks.
// IMPORTANT: This should be conservative - returns true only for truly trivial requests.
// If there's any mention of tools, functions, or writing, it's NOT background.
func hasBackgroundPattern(messages []MessageContent) bool {
	// If ANY tool keywords appear, it's NOT a background task
	toolBlockers := []string{
		"tool", "function", "execute", "run command",
		"write", "edit", "create", "delete", "remove",
		"implement", "build", "add", "modify",
	}

	for _, msg := range messages {
		lower := strings.ToLower(msg.Content)
		for _, kw := range toolBlockers {
			if strings.Contains(lower, kw) {
				return false
			}
		}
	}

	// Only truly simple operations are background tasks
	backgroundKeywords := []string{
		"list directory", "ls -", "dir",
		"show file", "view file", "cat file",
		"what is", "what's", "tell me about",
		"check status", "show status",
	}

	for _, msg := range messages {
		lower := strings.ToLower(msg.Content)
		for _, kw := range backgroundKeywords {
			if strings.Contains(lower, kw) {
				return true
			}
		}
	}
	return false
}

// getLongContextThreshold returns the configured threshold or a sensible default.
// Default is 100K tokens to trigger long-context models (1M context) vs regular models (128-256K).
func getLongContextThreshold(cfg *config.Config) int {
	if cfg != nil {
		if lc, ok := cfg.Models["long_context"]; ok && lc.ContextThreshold > 0 {
			return lc.ContextThreshold
		}
	}
	return 100000 // Default: 100K tokens
}

func scenarioModelName(cfg *config.Config, scenario Scenario) string {
	if cfg != nil {
		if model, ok := cfg.Models[string(scenario)]; ok && model.ModelID != "" {
			return model.ModelID
		}
	}
	return string(scenario)
}

// RouteForStreaming selects a model optimized for streaming latency.
// For streaming, we prioritize fast TTFT (time-to-first-token) over capability.
// This may return a less capable model but one that streams faster.
func RouteForStreaming(messages []MessageContent, tokenCount int, cfg *config.Config) ScenarioResult {
	// For streaming, use simpler models that have better TTFT
	// Complex models can be too slow for streaming with many tools.

	threshold := getLongContextThreshold(cfg)
	if tokenCount > threshold {
		model := scenarioModelName(cfg, ScenarioLongContext)
		return ScenarioResult{
			Scenario:   ScenarioLongContext,
			TokenCount: tokenCount,
			Reason:     fmt.Sprintf("high token count streaming (%d > %d) - use %s for acceptable TTFT", tokenCount, threshold, model),
		}
	}

	if hasComplexPattern(messages) || hasThinkingPattern(messages) {
		// Complex request but streaming - downgrade to faster model
		// Prefer the configured fast model for lower time-to-first-token.
		return ScenarioResult{
			Scenario:   ScenarioFast,
			TokenCount: tokenCount,
			Reason:     fmt.Sprintf("complex request but streaming - use %s for better TTFT", scenarioModelName(cfg, ScenarioFast)),
		}
	}

	// Default to fast scenario for streaming
	return ScenarioResult{
		Scenario:   ScenarioFast,
		TokenCount: tokenCount,
		Reason:     fmt.Sprintf("streaming request - use %s", scenarioModelName(cfg, ScenarioFast)),
	}
}
