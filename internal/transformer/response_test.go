package transformer

import (
	"testing"

	"oc-go-cc/pkg/types"
)

func TestTransformResponsePreservesReasoningContent(t *testing.T) {
	transformer := NewResponseTransformer()

	reasoning := "Let me think about this step by step"
	resp := &types.ChatCompletionResponse{
		ID:      "chatcmpl_123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "kimi-k2.6",
		Choices: []types.Choice{
			{
				Index: 0,
				Message: types.ChatMessage{
					Role:             "assistant",
					Content:          "The answer is 42.",
					ReasoningContent: &reasoning,
				},
				FinishReason: "stop",
			},
		},
		Usage: types.UsageInfo{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	anthropicResp, err := transformer.TransformResponse(resp, "kimi-k2.6")
	if err != nil {
		t.Fatalf("TransformResponse() error = %v", err)
	}

	if got, want := len(anthropicResp.Content), 2; got != want {
		t.Fatalf("len(Content) = %d, want %d", got, want)
	}

	if got, want := anthropicResp.Content[0].Type, "thinking"; got != want {
		t.Fatalf("Content[0].Type = %q, want %q", got, want)
	}
	if got, want := anthropicResp.Content[0].Thinking, reasoning; got != want {
		t.Fatalf("Content[0].Thinking = %q, want %q", got, want)
	}

	if got, want := anthropicResp.Content[1].Type, "text"; got != want {
		t.Fatalf("Content[1].Type = %q, want %q", got, want)
	}
	if got, want := anthropicResp.Content[1].Text, "The answer is 42."; got != want {
		t.Fatalf("Content[1].Text = %q, want %q", got, want)
	}
}

func TestTransformResponsePreservesReasoningContentWithToolCalls(t *testing.T) {
	transformer := NewResponseTransformer()

	reasoning := "I need to call a tool to get the weather"
	resp := &types.ChatCompletionResponse{
		ID:      "chatcmpl_456",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "kimi-k2.6",
		Choices: []types.Choice{
			{
				Index: 0,
				Message: types.ChatMessage{
					Role:             "assistant",
					Content:          "",
					ReasoningContent: &reasoning,
					ToolCalls: []types.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: types.FunctionCall{
								Name:      "get_weather",
								Arguments: `{"city":"Kigali"}`,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
		Usage: types.UsageInfo{
			PromptTokens:     20,
			CompletionTokens: 15,
			TotalTokens:      35,
		},
	}

	anthropicResp, err := transformer.TransformResponse(resp, "kimi-k2.6")
	if err != nil {
		t.Fatalf("TransformResponse() error = %v", err)
	}

	if got, want := len(anthropicResp.Content), 2; got != want {
		t.Fatalf("len(Content) = %d, want %d", got, want)
	}

	if got, want := anthropicResp.Content[0].Type, "thinking"; got != want {
		t.Fatalf("Content[0].Type = %q, want %q", got, want)
	}
	if got, want := anthropicResp.Content[0].Thinking, reasoning; got != want {
		t.Fatalf("Content[0].Thinking = %q, want %q", got, want)
	}

	if got, want := anthropicResp.Content[1].Type, "tool_use"; got != want {
		t.Fatalf("Content[1].Type = %q, want %q", got, want)
	}
	if got, want := anthropicResp.Content[1].Name, "get_weather"; got != want {
		t.Fatalf("Content[1].Name = %q, want %q", got, want)
	}

	if got, want := anthropicResp.StopReason, "tool_use"; got != want {
		t.Fatalf("StopReason = %q, want %q", got, want)
	}
}

func TestTransformResponseOmitsEmptyReasoningContent(t *testing.T) {
	transformer := NewResponseTransformer()

	emptyReasoning := ""
	resp := &types.ChatCompletionResponse{
		ID:      "chatcmpl_789",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "kimi-k2.6",
		Choices: []types.Choice{
			{
				Index: 0,
				Message: types.ChatMessage{
					Role:             "assistant",
					Content:          "Hello there.",
					ReasoningContent: &emptyReasoning,
				},
				FinishReason: "stop",
			},
		},
		Usage: types.UsageInfo{
			PromptTokens:     5,
			CompletionTokens: 2,
			TotalTokens:      7,
		},
	}

	anthropicResp, err := transformer.TransformResponse(resp, "kimi-k2.6")
	if err != nil {
		t.Fatalf("TransformResponse() error = %v", err)
	}

	if got, want := len(anthropicResp.Content), 1; got != want {
		t.Fatalf("len(Content) = %d, want %d", got, want)
	}

	if got, want := anthropicResp.Content[0].Type, "text"; got != want {
		t.Fatalf("Content[0].Type = %q, want %q", got, want)
	}
}

func TestTransformResponseNoReasoningContent(t *testing.T) {
	transformer := NewResponseTransformer()

	resp := &types.ChatCompletionResponse{
		ID:      "chatcmpl_abc",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "kimi-k2.6",
		Choices: []types.Choice{
			{
				Index: 0,
				Message: types.ChatMessage{
					Role:    "assistant",
					Content: "Just a plain response.",
				},
				FinishReason: "stop",
			},
		},
		Usage: types.UsageInfo{
			PromptTokens:     3,
			CompletionTokens: 4,
			TotalTokens:      7,
		},
	}

	anthropicResp, err := transformer.TransformResponse(resp, "kimi-k2.6")
	if err != nil {
		t.Fatalf("TransformResponse() error = %v", err)
	}

	if got, want := len(anthropicResp.Content), 1; got != want {
		t.Fatalf("len(Content) = %d, want %d", got, want)
	}

	if got, want := anthropicResp.Content[0].Type, "text"; got != want {
		t.Fatalf("Content[0].Type = %q, want %q", got, want)
	}
}
