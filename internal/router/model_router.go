// Package router defines HTTP route registration and middleware chaining,
// as well as model selection based on request scenarios.
package router

import (
	"fmt"

	"oc-go-cc/internal/config"
)

// ModelRouter handles model selection based on scenarios.
type ModelRouter struct {
	config *config.Config
}

// NewModelRouter creates a new model router.
func NewModelRouter(cfg *config.Config) *ModelRouter {
	return &ModelRouter{config: cfg}
}

// RouteResult contains the selected model and fallback chain.
type RouteResult struct {
	Primary   config.ModelConfig
	Fallbacks []config.ModelConfig
	Scenario  Scenario
}

// Route determines which model to use for a request.
func (r *ModelRouter) Route(messages []MessageContent, tokenCount int) (RouteResult, error) {
	result := DetectScenario(messages, tokenCount, r.config)

	// Get primary model for scenario
	primary, ok := r.config.Models[string(result.Scenario)]
	if !ok {
		// Fall back to default if scenario model not configured
		primary, ok = r.config.Models["default"]
		if !ok {
			return RouteResult{}, fmt.Errorf("no default model configured")
		}
	}

	// Get fallbacks for scenario
	fallbacks := r.config.Fallbacks[string(result.Scenario)]
	if len(fallbacks) == 0 {
		// Fall back to default fallbacks
		fallbacks = r.config.Fallbacks["default"]
	}
	fallbacks = r.resolveModelConfigs(fallbacks)

	return RouteResult{
		Primary:   primary,
		Fallbacks: fallbacks,
		Scenario:  result.Scenario,
	}, nil
}

// GetModelChain returns the full chain of models to try (primary + fallbacks).
func (rr *RouteResult) GetModelChain() []config.ModelConfig {
	seen := make(map[string]bool, len(rr.Fallbacks)+1)
	chain := make([]config.ModelConfig, 0, len(rr.Fallbacks)+1)

	for _, model := range append([]config.ModelConfig{rr.Primary}, rr.Fallbacks...) {
		if model.ModelID == "" || seen[model.ModelID] {
			continue
		}
		seen[model.ModelID] = true
		chain = append(chain, model)
	}
	return chain
}

// RouteForStreaming determines which model to use for streaming requests.
// Preserves capability for complex/thinking/long-context requests.
func (r *ModelRouter) RouteForStreaming(messages []MessageContent, tokenCount int) RouteResult {
	result := RouteForStreaming(messages, tokenCount, r.config)

	// Get primary model for scenario
	primary, ok := r.config.Models[string(result.Scenario)]
	if !ok {
		// Fall back to fast scenario if not configured
		primary, ok = r.config.Models["fast"]
		if !ok {
			// Fall back to default
			primary = r.config.Models["default"]
		}
	}

	// Get fallbacks for scenario
	fallbacks := r.config.Fallbacks[string(result.Scenario)]
	if len(fallbacks) == 0 {
		// Fall back to fast fallbacks
		fallbacks = r.config.Fallbacks["fast"]
	}
	fallbacks = r.resolveModelConfigs(fallbacks)

	return RouteResult{
		Primary:   primary,
		Fallbacks: fallbacks,
		Scenario:  result.Scenario,
	}
}

func (r *ModelRouter) resolveModelConfigs(models []config.ModelConfig) []config.ModelConfig {
	if r == nil || r.config == nil || len(models) == 0 {
		return models
	}

	result := make([]config.ModelConfig, 0, len(models))
	for _, model := range models {
		result = append(result, r.resolveModelConfig(model))
	}
	return result
}

func (r *ModelRouter) resolveModelConfig(model config.ModelConfig) config.ModelConfig {
	if r == nil || r.config == nil {
		return model
	}

	if registered, ok := r.config.Models[model.ModelID]; ok && registered.ModelID == model.ModelID {
		return mergeModelConfig(registered, model)
	}

	for _, registered := range r.config.Models {
		if registered.ModelID == "" || registered.ModelID != model.ModelID {
			continue
		}
		return mergeModelConfig(registered, model)
	}
	return model
}

func mergeModelConfig(base, override config.ModelConfig) config.ModelConfig {
	merged := base
	if override.Provider != "" {
		merged.Provider = override.Provider
	}
	if override.ModelID != "" {
		merged.ModelID = override.ModelID
	}
	if override.EndpointType != "" {
		merged.EndpointType = override.EndpointType
	}
	if override.Temperature > 0 {
		merged.Temperature = override.Temperature
	}
	if override.MaxTokens > 0 {
		merged.MaxTokens = override.MaxTokens
	}
	if override.ContextThreshold > 0 {
		merged.ContextThreshold = override.ContextThreshold
	}
	if override.ReasoningEffort != "" {
		merged.ReasoningEffort = override.ReasoningEffort
	}
	if override.ReasoningFormat != "" {
		merged.ReasoningFormat = override.ReasoningFormat
	}
	if override.SupportsThinking != nil {
		merged.SupportsThinking = override.SupportsThinking
	}
	if override.RequiresReasoningContent != nil {
		merged.RequiresReasoningContent = override.RequiresReasoningContent
	}
	return merged
}
