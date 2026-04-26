// Package transformer handles request and response format conversion
// between Anthropic Messages API and OpenAI Chat Completions API.
package transformer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"oc-go-cc/internal/config"
	"oc-go-cc/pkg/types"
)

// RequestTransformer converts Anthropic requests to OpenAI format.
type RequestTransformer struct{}

// NewRequestTransformer creates a new request transformer.
func NewRequestTransformer() *RequestTransformer {
	return &RequestTransformer{}
}

// TransformRequest converts an Anthropic MessageRequest to OpenAI ChatCompletionRequest.
func (t *RequestTransformer) TransformRequest(
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
) (*types.ChatCompletionRequest, error) {
	// Transform messages
	messages, err := t.transformMessages(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to transform messages: %w", err)
	}

	// Build OpenAI request
	openaiReq := &types.ChatCompletionRequest{
		Model:    model.ModelID,
		Messages: messages,
		Stream:   anthropicReq.Stream,
	}

	// Copy optional parameters from Anthropic request
	if anthropicReq.Temperature != nil {
		openaiReq.Temperature = anthropicReq.Temperature
	}
	if anthropicReq.TopP != nil {
		openaiReq.TopP = anthropicReq.TopP
	}

	// Map max_tokens
	if anthropicReq.MaxTokens > 0 {
		maxTokens := anthropicReq.MaxTokens
		openaiReq.MaxTokens = &maxTokens
	}

	// Apply model-specific overrides
	if model.Temperature > 0 {
		openaiReq.Temperature = &model.Temperature
	}
	if model.MaxTokens > 0 {
		maxTokens := model.MaxTokens
		openaiReq.MaxTokens = &maxTokens
	}

	// Transform tools if present
	if len(anthropicReq.Tools) > 0 {
		openaiReq.Tools = t.transformTools(anthropicReq.Tools)
	}

	t.applyDeepSeekReasoningOptions(openaiReq, anthropicReq, model)

	return openaiReq, nil
}

// applyDeepSeekReasoningOptions maps Claude/Anthropic effort controls to
// DeepSeek's OpenAI-compatible thinking fields.
func (t *RequestTransformer) applyDeepSeekReasoningOptions(
	openaiReq *types.ChatCompletionRequest,
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
) {
	if !isDeepSeekModel(model.ModelID) {
		return
	}

	effort := resolveDeepSeekReasoningEffort(anthropicReq, model)
	if effort == "" && !isThinkingEnabled(anthropicReq.Thinking) {
		return
	}

	openaiReq.Thinking = json.RawMessage(`{"type":"enabled"}`)
	if effort != "" {
		openaiReq.ReasoningEffort = effort
	}
}

func isDeepSeekModel(modelID string) bool {
	return strings.HasPrefix(strings.ToLower(modelID), "deepseek-")
}

func resolveDeepSeekReasoningEffort(anthropicReq *types.MessageRequest, model config.ModelConfig) string {
	candidates := []string{
		model.ReasoningEffort,
	}
	if anthropicReq.OutputConfig != nil {
		candidates = append(candidates, anthropicReq.OutputConfig.Effort)
	}
	candidates = append(candidates,
		os.Getenv("OC_GO_CC_REASONING_EFFORT"),
		os.Getenv("CLAUDE_CODE_EFFORT_LEVEL"),
	)

	for _, candidate := range candidates {
		if effort := normalizeDeepSeekReasoningEffort(candidate); effort != "" {
			return effort
		}
	}
	return ""
}

func normalizeDeepSeekReasoningEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "max", "xhigh":
		return "max"
	case "high", "medium", "low":
		return "high"
	default:
		return ""
	}
}

func isThinkingEnabled(raw json.RawMessage) bool {
	if strings.TrimSpace(string(raw)) == "" || strings.TrimSpace(string(raw)) == "null" {
		return false
	}

	var cfg struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return true
	}
	return strings.ToLower(cfg.Type) != "disabled"
}

// transformMessages converts Anthropic messages to OpenAI format.
func (t *RequestTransformer) transformMessages(anthropicReq *types.MessageRequest) ([]types.ChatMessage, error) {
	var result []types.ChatMessage

	// Add system message if present, preserving cache_control if available
	systemText := anthropicReq.SystemText()
	if systemText != "" {
		systemMsg := types.ChatMessage{
			Role:    "system",
			Content: systemText,
		}
		// Try to extract cache_control from system array blocks
		if len(anthropicReq.System) > 0 {
			var blocks []types.SystemContentBlock
			if err := json.Unmarshal(anthropicReq.System, &blocks); err == nil {
				for _, b := range blocks {
					if b.Type == "text" && b.CacheControl != nil {
						systemMsg.CacheControl = b.CacheControl
						break
					}
				}
			}
		}
		result = append(result, systemMsg)
	}

	// Transform each message
	for _, msg := range anthropicReq.Messages {
		openaiMsgs, err := t.transformMessage(msg)
		if err != nil {
			return nil, err
		}
		result = append(result, openaiMsgs...)
	}

	return result, nil
}

// transformMessage converts a single Anthropic message to one or more OpenAI messages.
// Tool_use and tool_result require special handling to map to OpenAI's function calling format.
func (t *RequestTransformer) transformMessage(msg types.Message) ([]types.ChatMessage, error) {
	blocks := msg.ContentBlocks()

	switch msg.Role {
	case "user":
		return t.transformUserMessage(blocks)
	case "assistant":
		return t.transformAssistantMessage(blocks)
	default:
		// Fallback: concatenate all text
		var text string
		for _, b := range blocks {
			if b.Type == "text" {
				text += b.Text
			}
		}
		return []types.ChatMessage{{Role: msg.Role, Content: text}}, nil
	}
}

// transformUserMessage converts a user message with potential tool_result blocks.
func (t *RequestTransformer) transformUserMessage(blocks []types.ContentBlock) ([]types.ChatMessage, error) {
	var result []types.ChatMessage
	var textParts []string

	for _, block := range blocks {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "tool_result":
			// In OpenAI, tool results are separate messages with role "tool"
			toolContent := block.TextContent()
			result = append(result, types.ChatMessage{
				Role:       "tool",
				Content:    toolContent,
				ToolCallID: block.GetToolID(),
			})
		case "image":
			// Images not supported in text-only models, skip
			textParts = append(textParts, "[Image]")
		}
	}

	// If there's text content, add it as a user message
	if len(textParts) > 0 {
		text := ""
		for _, p := range textParts {
			text += p
		}
		// OpenAI-compatible tool calling requires tool responses to appear
		// immediately after the assistant message that emitted tool_calls.
		// If the Anthropic user turn also includes free-form text, emit it as
		// a subsequent user message after all tool results.
		userMsg := types.ChatMessage{Role: "user", Content: text}
		result = append(result, userMsg)
	}

	return result, nil
}

// transformAssistantMessage converts an assistant message with potential tool_use blocks.
func (t *RequestTransformer) transformAssistantMessage(blocks []types.ContentBlock) ([]types.ChatMessage, error) {
	var textParts []string
	var thinkingParts []string
	var toolCalls []types.ToolCall

	for _, block := range blocks {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "thinking":
			// Some OpenAI-compatible providers expect tool-call assistant messages
			// to preserve chain-of-thought in a provider-specific reasoning field.
			if block.Thinking != "" {
				thinkingParts = append(thinkingParts, block.Thinking)
			}
		case "tool_use":
			// Map to OpenAI function call format
			arguments := "{}"
			if len(block.Input) > 0 {
				arguments = string(block.Input)
			}
			toolCalls = append(toolCalls, types.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: types.FunctionCall{
					Name:      block.Name,
					Arguments: arguments,
				},
			})
		}
	}

	// Build the assistant message
	content := ""
	for _, p := range textParts {
		content += p
	}
	reasoningContent := ""
	for _, p := range thinkingParts {
		reasoningContent += p
	}

	var reasoningContentPtr *string
	if len(toolCalls) > 0 || reasoningContent != "" {
		// Some providers require reasoning_content to be present on assistant
		// tool-call messages whenever thinking mode is enabled, even if the
		// upstream Anthropic history did not include an explicit thinking block.
		// Moonshot's validator treats an empty string as missing, so use a
		// non-empty placeholder when we must provide the field.
		if reasoningContent == "" {
			reasoningContent = " "
		}
		reasoningContentPtr = &reasoningContent
	}

	msg := types.ChatMessage{
		Role:             "assistant",
		Content:          content,
		ReasoningContent: reasoningContentPtr,
		ToolCalls:        toolCalls,
	}

	return []types.ChatMessage{msg}, nil
}

// transformTools converts Anthropic tools to OpenAI tools.
func (t *RequestTransformer) transformTools(tools []types.Tool) []types.ToolDef {
	var result []types.ToolDef

	for _, tool := range tools {
		// InputSchema is already json.RawMessage, use it directly
		schema := tool.InputSchema
		if len(schema) == 0 {
			schema = []byte(`{"type":"object","properties":{}}`)
		}

		result = append(result, types.ToolDef{
			Type: "function",
			Function: types.FunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  json.RawMessage(schema),
			},
		})
	}

	return result
}
