package config

import "testing"

func TestModelConfigEndpointTypeResolution(t *testing.T) {
	tests := []struct {
		name string
		cfg  ModelConfig
		want string
	}{
		{
			name: "explicit anthropic",
			cfg:  ModelConfig{ModelID: "deepseek-v4-pro", EndpointType: "anthropic"},
			want: EndpointTypeAnthropic,
		},
		{
			name: "explicit openai",
			cfg:  ModelConfig{ModelID: "minimax-m2.7", EndpointType: "openai"},
			want: EndpointTypeOpenAI,
		},
		{
			name: "known minimax default",
			cfg:  ModelConfig{ModelID: "minimax-m2.7"},
			want: EndpointTypeAnthropic,
		},
		{
			name: "deepseek default",
			cfg:  ModelConfig{ModelID: "deepseek-v4-pro"},
			want: EndpointTypeOpenAI,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.cfg.EffectiveEndpointType(); got != tt.want {
				t.Fatalf("EffectiveEndpointType() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestModelConfigThinkingDefaultsAndOverrides(t *testing.T) {
	if !((&ModelConfig{ModelID: "deepseek-v4-pro"}).EffectiveSupportsThinking()) {
		t.Fatal("deepseek should default to supports thinking")
	}
	if !((&ModelConfig{ModelID: "deepseek-v4-pro"}).EffectiveRequiresReasoningContent()) {
		t.Fatal("deepseek should default to requiring reasoning_content")
	}

	disabled := false
	cfg := ModelConfig{
		ModelID:                  "deepseek-v4-pro",
		SupportsThinking:         &disabled,
		RequiresReasoningContent: &disabled,
	}
	if cfg.EffectiveSupportsThinking() {
		t.Fatal("explicit supports_thinking=false was ignored")
	}
	if cfg.EffectiveRequiresReasoningContent() {
		t.Fatal("explicit requires_reasoning_content=false was ignored")
	}
}
