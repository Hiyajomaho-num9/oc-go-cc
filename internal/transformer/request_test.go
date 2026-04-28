package transformer

import (
	"encoding/json"
	"testing"

	"oc-go-cc/internal/config"
	"oc-go-cc/pkg/types"
)

func TestTransformRequestPreservesThinkingAsReasoningContent(t *testing.T) {
	transformer := NewRequestTransformer()
	stream := true

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Stream:    &stream,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"thinking","thinking":"Need to look up the weather first","signature":"sig_123"},
					{"type":"tool_use","id":"toolu_123","name":"get_weather","input":{"city":"Kigali"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := len(openaiReq.Messages), 1; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}

	msg := openaiReq.Messages[0]
	if got, want := msg.Role, "assistant"; got != want {
		t.Fatalf("Role = %q, want %q", got, want)
	}
	if msg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil")
	}
	if got, want := *msg.ReasoningContent, "Need to look up the weather first"; got != want {
		t.Fatalf("ReasoningContent = %q, want %q", got, want)
	}
	if got, want := len(msg.ToolCalls), 1; got != want {
		t.Fatalf("len(ToolCalls) = %d, want %d", got, want)
	}
	if got, want := msg.ToolCalls[0].ID, "toolu_123"; got != want {
		t.Fatalf("ToolCalls[0].ID = %q, want %q", got, want)
	}
	if got, want := msg.ToolCalls[0].Function.Name, "get_weather"; got != want {
		t.Fatalf("ToolCalls[0].Function.Name = %q, want %q", got, want)
	}
	if got, want := msg.ToolCalls[0].Function.Arguments, `{"city":"Kigali"}`; got != want {
		t.Fatalf("ToolCalls[0].Function.Arguments = %q, want %q", got, want)
	}
}

func TestTransformRequestIncludesStreamUsageOptions(t *testing.T) {
	transformer := NewRequestTransformer()
	stream := true

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Stream:    &stream,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.StreamOptions == nil {
		t.Fatal("StreamOptions = nil, want include_usage enabled")
	}
	if !openaiReq.StreamOptions.IncludeUsage {
		t.Fatal("StreamOptions.IncludeUsage = false, want true")
	}
}

func TestTransformRequestIncludesEmptyReasoningContentForToolCalls(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_456","name":"search_docs","input":{"query":"figma api"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	msg := openaiReq.Messages[0]
	if msg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder")
	}
	if got, want := *msg.ReasoningContent, " "; got != want {
		t.Fatalf("ReasoningContent = %q, want %q", got, want)
	}
}

func TestTransformRequestIncludesPlaceholderReasoningForDeepSeekToolCalls(t *testing.T) {
	t.Setenv("OC_GO_CC_REASONING_EFFORT", "")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")

	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_789","name":"read_file","input":{"path":"README.md"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	msg := openaiReq.Messages[0]
	if msg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for DeepSeek tool call")
	}
	if got, want := *msg.ReasoningContent, " "; got != want {
		t.Fatalf("ReasoningContent = %q, want %q", got, want)
	}
}

func TestTransformRequestIncludesPlaceholderReasoningForDeepSeekThinkingTextAssistant(t *testing.T) {
	t.Setenv("OC_GO_CC_REASONING_EFFORT", "max")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")

	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role:    "assistant",
				Content: json.RawMessage(`[{"type":"text","text":"I checked the build log."}]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	msg := openaiReq.Messages[0]
	if msg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for DeepSeek thinking history")
	}
	if got, want := *msg.ReasoningContent, " "; got != want {
		t.Fatalf("ReasoningContent = %q, want %q", got, want)
	}
}

func TestTransformRequestDoesNotAddPlaceholderReasoningForDeepSeekTextAssistantWithoutThinking(t *testing.T) {
	t.Setenv("OC_GO_CC_REASONING_EFFORT", "")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")

	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role:    "assistant",
				Content: json.RawMessage(`[{"type":"text","text":"Plain assistant answer."}]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got := openaiReq.Messages[0].ReasoningContent; got != nil {
		t.Fatalf("ReasoningContent = %q, want nil without DeepSeek thinking mode", *got)
	}
}

func TestTransformRequestDoesNotAddPlaceholderReasoningForGenericToolCalls(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_generic","name":"search","input":{"query":"docs"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "qwen3.6-plus"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got := openaiReq.Messages[0].ReasoningContent; got != nil {
		t.Fatalf("ReasoningContent = %q, want nil for generic model", *got)
	}
}

func TestTransformRequestPreservesSystemCacheControl(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		System: json.RawMessage(`[
			{"type":"text","text":"You are helpful","cache_control":{"type":"ephemeral"}}
		]`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := len(openaiReq.Messages), 2; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}

	systemMsg := openaiReq.Messages[0]
	if got, want := systemMsg.Role, "system"; got != want {
		t.Fatalf("Messages[0].Role = %q, want %q", got, want)
	}
	if got, want := systemMsg.Content, "You are helpful"; got != want {
		t.Fatalf("Messages[0].Content = %q, want %q", got, want)
	}
	if systemMsg.CacheControl == nil {
		t.Fatal("Messages[0].CacheControl = nil, want non-nil")
	}
	if got, want := systemMsg.CacheControl.Type, "ephemeral"; got != want {
		t.Fatalf("Messages[0].CacheControl.Type = %q, want %q", got, want)
	}
}

func TestTransformRequestOmitsCacheControlWhenAbsent(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		System:    json.RawMessage(`"You are helpful"`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := len(openaiReq.Messages), 2; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}

	systemMsg := openaiReq.Messages[0]
	if got, want := systemMsg.Role, "system"; got != want {
		t.Fatalf("Messages[0].Role = %q, want %q", got, want)
	}
	if systemMsg.CacheControl != nil {
		t.Fatalf("Messages[0].CacheControl = %v, want nil", systemMsg.CacheControl)
	}
}

func TestTransformRequestPlacesToolResultsBeforeUserText(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_123","name":"create_file","input":{"name":"draft.fig"}}
				]`),
			},
			{
				Role: "user",
				Content: json.RawMessage(`[
					{"type":"tool_result","tool_use_id":"toolu_123","content":"created"},
					{"type":"text","text":"now continue"}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := len(openaiReq.Messages), 3; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}

	if got, want := openaiReq.Messages[0].Role, "assistant"; got != want {
		t.Fatalf("Messages[0].Role = %q, want %q", got, want)
	}
	if got, want := openaiReq.Messages[1].Role, "tool"; got != want {
		t.Fatalf("Messages[1].Role = %q, want %q", got, want)
	}
	if got, want := openaiReq.Messages[1].ToolCallID, "toolu_123"; got != want {
		t.Fatalf("Messages[1].ToolCallID = %q, want %q", got, want)
	}
	if got, want := openaiReq.Messages[2].Role, "user"; got != want {
		t.Fatalf("Messages[2].Role = %q, want %q", got, want)
	}
	if got, want := openaiReq.Messages[2].Content, "now continue"; got != want {
		t.Fatalf("Messages[2].Content = %q, want %q", got, want)
	}
}

func TestTransformRequestMapsOutputConfigEffortForDeepSeek(t *testing.T) {
	t.Setenv("OC_GO_CC_REASONING_EFFORT", "")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")

	transformer := NewRequestTransformer()
	req := &types.MessageRequest{
		Model:        "claude-test",
		MaxTokens:    256,
		Thinking:     json.RawMessage(`{"type":"adaptive"}`),
		OutputConfig: &types.OutputConfig{Effort: "max"},
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := string(openaiReq.Thinking), `{"type":"enabled"}`; got != want {
		t.Fatalf("Thinking = %s, want %s", got, want)
	}
	if got, want := openaiReq.ReasoningEffort, "max"; got != want {
		t.Fatalf("ReasoningEffort = %q, want %q", got, want)
	}
}

func TestTransformRequestUsesConfiguredReasoningEffortForDeepSeek(t *testing.T) {
	t.Setenv("OC_GO_CC_REASONING_EFFORT", "")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")

	transformer := NewRequestTransformer()
	req := &types.MessageRequest{
		Model:        "claude-test",
		MaxTokens:    256,
		OutputConfig: &types.OutputConfig{Effort: "high"},
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "deepseek-v4-flash",
		ReasoningEffort: "xhigh",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := openaiReq.ReasoningEffort, "max"; got != want {
		t.Fatalf("ReasoningEffort = %q, want %q", got, want)
	}
}

func TestTransformRequestUsesEnvReasoningEffortForDeepSeek(t *testing.T) {
	t.Setenv("OC_GO_CC_REASONING_EFFORT", "max")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")

	transformer := NewRequestTransformer()
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := openaiReq.ReasoningEffort, "max"; got != want {
		t.Fatalf("ReasoningEffort = %q, want %q", got, want)
	}
	if len(openaiReq.Thinking) == 0 {
		t.Fatal("Thinking is empty, want enabled thinking when env effort is set")
	}
}

func TestTransformRequestDoesNotApplyDeepSeekReasoningToOtherModels(t *testing.T) {
	t.Setenv("OC_GO_CC_REASONING_EFFORT", "max")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "max")

	transformer := NewRequestTransformer()
	req := &types.MessageRequest{
		Model:        "claude-test",
		MaxTokens:    256,
		Thinking:     json.RawMessage(`{"type":"adaptive"}`),
		OutputConfig: &types.OutputConfig{Effort: "max"},
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if len(openaiReq.Thinking) != 0 {
		t.Fatalf("Thinking = %s, want empty for non-DeepSeek model", string(openaiReq.Thinking))
	}
	if openaiReq.ReasoningEffort != "" {
		t.Fatalf("ReasoningEffort = %q, want empty for non-DeepSeek model", openaiReq.ReasoningEffort)
	}
}

func TestTransformRequestMapsStopSequencesAndToolChoice(t *testing.T) {
	transformer := NewRequestTransformer()
	req := &types.MessageRequest{
		Model:         "claude-test",
		MaxTokens:     256,
		StopSequences: []string{"STOP"},
		ToolChoice:    json.RawMessage(`{"type":"tool","name":"read_file"}`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	stop, ok := openaiReq.Stop.([]string)
	if !ok || len(stop) != 1 || stop[0] != "STOP" {
		t.Fatalf("Stop = %#v, want STOP slice", openaiReq.Stop)
	}

	choice, ok := openaiReq.ToolChoice.(map[string]interface{})
	if !ok {
		t.Fatalf("ToolChoice = %#v, want OpenAI function choice", openaiReq.ToolChoice)
	}
	fn := choice["function"].(map[string]interface{})
	if got, want := fn["name"], "read_file"; got != want {
		t.Fatalf("tool choice name = %q, want %q", got, want)
	}
}

func TestTransformRequestSupportsConfiguredThinkingModel(t *testing.T) {
	t.Setenv("OC_GO_CC_REASONING_EFFORT", "")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")
	supportsThinking := true

	transformer := NewRequestTransformer()
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Thinking:  json.RawMessage(`{"type":"enabled"}`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:          "custom-reasoner",
		SupportsThinking: &supportsThinking,
		ReasoningFormat:  "openai",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}
	if len(openaiReq.Thinking) == 0 {
		t.Fatal("Thinking is empty, want enabled thinking for configured reasoner")
	}
}
