// Package config handles application configuration loading and validation.
package config

import "strings"

// Config holds the complete application configuration.
type Config struct {
	APIKey     string                   `json:"api_key"`
	Host       string                   `json:"host"`
	Port       int                      `json:"port"`
	Models     map[string]ModelConfig   `json:"models"`
	Fallbacks  map[string][]ModelConfig `json:"fallbacks"`
	OpenCodeGo OpenCodeGoConfig         `json:"opencode_go"`
	Logging    LoggingConfig            `json:"logging"`
}

// ModelConfig defines routing rules for a specific model.
type ModelConfig struct {
	Provider                 string  `json:"provider"`
	ModelID                  string  `json:"model_id"`
	EndpointType             string  `json:"endpoint_type,omitempty"`
	Temperature              float64 `json:"temperature"`
	MaxTokens                int     `json:"max_tokens"`
	ContextThreshold         int     `json:"context_threshold"`
	ReasoningEffort          string  `json:"reasoning_effort,omitempty"`
	ReasoningFormat          string  `json:"reasoning_format,omitempty"`
	SupportsThinking         *bool   `json:"supports_thinking,omitempty"`
	RequiresReasoningContent *bool   `json:"requires_reasoning_content,omitempty"`
}

// OpenCodeGoConfig holds the upstream OpenCode Go API settings.
type OpenCodeGoConfig struct {
	BaseURL          string `json:"base_url"`
	AnthropicBaseURL string `json:"anthropic_base_url"`
	TimeoutMs        int    `json:"timeout_ms"`
	StreamTimeoutMs  int    `json:"stream_timeout_ms"`
}

// LoggingConfig controls application logging behavior.
type LoggingConfig struct {
	Level    string `json:"level"`
	Requests bool   `json:"requests"`
}

const (
	EndpointTypeOpenAI    = "openai"
	EndpointTypeAnthropic = "anthropic"
)

// EffectiveEndpointType resolves the provider endpoint used by this model.
// Unknown values default to OpenAI-compatible because most OpenCode Go models
// expose chat/completions.
func (m ModelConfig) EffectiveEndpointType() string {
	switch strings.ToLower(strings.TrimSpace(m.EndpointType)) {
	case EndpointTypeAnthropic, "anthropic-compatible", "messages", "/v1/messages":
		return EndpointTypeAnthropic
	case EndpointTypeOpenAI, "openai-compatible", "chat_completions", "chat-completions", "/v1/chat/completions":
		return EndpointTypeOpenAI
	}

	switch strings.ToLower(strings.TrimSpace(m.ModelID)) {
	case "minimax-m2.5", "minimax-m2.7":
		return EndpointTypeAnthropic
	default:
		return EndpointTypeOpenAI
	}
}

// UsesAnthropicEndpoint reports whether the model should skip OpenAI
// transformation and be sent to the Anthropic-compatible upstream endpoint.
func (m ModelConfig) UsesAnthropicEndpoint() bool {
	return m.EffectiveEndpointType() == EndpointTypeAnthropic
}

// EffectiveSupportsThinking resolves whether the model accepts OpenAI-style
// thinking/reasoning controls.
func (m ModelConfig) EffectiveSupportsThinking() bool {
	if m.SupportsThinking != nil {
		return *m.SupportsThinking
	}
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(m.ModelID)), "deepseek-")
}

// EffectiveRequiresReasoningContent resolves whether assistant history should
// carry reasoning_content when tool calls or thinking mode are active.
func (m ModelConfig) EffectiveRequiresReasoningContent() bool {
	if m.RequiresReasoningContent != nil {
		return *m.RequiresReasoningContent
	}
	modelID := strings.ToLower(strings.TrimSpace(m.ModelID))
	return strings.HasPrefix(modelID, "deepseek-") || strings.HasPrefix(modelID, "kimi-")
}

// EffectiveReasoningFormat returns the configured reasoning dialect. The
// current transformer supports the OpenAI-compatible DeepSeek fields.
func (m ModelConfig) EffectiveReasoningFormat() string {
	format := strings.ToLower(strings.TrimSpace(m.ReasoningFormat))
	if format == "" && m.EffectiveSupportsThinking() {
		return "openai"
	}
	return format
}
