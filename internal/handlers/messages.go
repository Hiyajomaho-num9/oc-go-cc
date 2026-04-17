// Package handlers contains HTTP request handlers for API endpoints.
package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"oc-go-cc/internal/client"
	"oc-go-cc/internal/config"
	"oc-go-cc/internal/metrics"
	"oc-go-cc/internal/middleware"
	"oc-go-cc/internal/router"
	"oc-go-cc/internal/token"
	"oc-go-cc/internal/transformer"
	"oc-go-cc/pkg/types"
)

// MessagesHandler handles /v1/messages requests.
type MessagesHandler struct {
	config              *config.Config
	client              *client.OpenCodeClient
	modelRouter         *router.ModelRouter
	fallbackHandler     *router.FallbackHandler
	requestTransformer  *transformer.RequestTransformer
	responseTransformer *transformer.ResponseTransformer
	streamHandler       *transformer.StreamHandler
	tokenCounter        *token.Counter
	logger              *slog.Logger
	rateLimiter         *middleware.RateLimiter
	requestDedup        *middleware.RequestDeduplicator
	requestIDGen        *middleware.RequestIDGenerator
	metrics             *metrics.Metrics
}

// responseWriter wraps http.ResponseWriter to track if headers were written.
type responseWriter struct {
	http.ResponseWriter
	wroteHeader bool
}

func (w *responseWriter) WriteHeader(code int) {
	if !w.wroteHeader {
		w.wroteHeader = true
		w.ResponseWriter.WriteHeader(code)
	}
}

func (w *responseWriter) Write(b []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(b)
}

// Flush implements http.Flusher for SSE streaming support.
func (w *responseWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// NewMessagesHandler creates a new messages handler.
func NewMessagesHandler(
	cfg *config.Config,
	openCodeClient *client.OpenCodeClient,
	modelRouter *router.ModelRouter,
	fallbackHandler *router.FallbackHandler,
	tokenCounter *token.Counter,
	metrics *metrics.Metrics,
) *MessagesHandler {
	return &MessagesHandler{
		config:              cfg,
		client:              openCodeClient,
		modelRouter:         modelRouter,
		fallbackHandler:     fallbackHandler,
		requestTransformer:  transformer.NewRequestTransformer(),
		responseTransformer: transformer.NewResponseTransformer(),
		streamHandler:       transformer.NewStreamHandler(),
		tokenCounter:        tokenCounter,
		logger:              slog.Default(),
		rateLimiter:         middleware.NewRateLimiter(100, time.Minute),
		requestDedup:        middleware.NewRequestDeduplicator(500 * time.Millisecond),
		requestIDGen:        middleware.NewRequestIDGenerator(),
		metrics:             metrics,
	}
}

// HandleMessages handles POST /v1/messages.
func (h *MessagesHandler) HandleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Generate or get request ID for correlation
	requestID := r.Header.Get("X-Request-ID")
	if requestID == "" {
		requestID = h.requestIDGen.Generate()
	}
	w.Header().Set("X-Request-ID", requestID)

	// Rate limiting
	clientIP := middleware.GetClientIP(r)
	if !h.rateLimiter.Allow(clientIP) {
		h.metrics.RecordRateLimited()
		h.logger.Warn("rate limited", "client", clientIP, "request_id", requestID)
		http.Error(w, "rate limited", http.StatusTooManyRequests)
		return
	}

	// Read the raw request body for debug logging
	var rawBody json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&rawBody); err != nil {
		h.sendError(w, http.StatusBadRequest, "invalid request body", err)
		return
	}

	// Deduplicate - skip duplicate requests
	if _, ok := h.requestDedup.TryAcquire(rawBody); !ok {
		h.metrics.RecordDeduplicated()
		h.logger.Info("duplicate request skipped", "request_id", requestID)
		return
	}

	// Parse into Anthropic request
	var anthropicReq types.MessageRequest
	if err := json.Unmarshal(rawBody, &anthropicReq); err != nil {
		h.sendError(w, http.StatusBadRequest, "invalid request body", err)
		return
	}

	// Validate request
	if err := anthropicReq.Validate(); err != nil {
		h.sendError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}

	// Record metrics
	isStreaming := anthropicReq.Stream != nil && *anthropicReq.Stream
	h.metrics.RecordRequest(isStreaming)

	h.logger.Info("received request",
		"model", anthropicReq.Model,
		"streaming", isStreaming,
		"messages", len(anthropicReq.Messages),
		"tools", len(anthropicReq.Tools),
		"max_tokens", anthropicReq.MaxTokens,
	)

	// Build message content for routing and token counting.
	var routerMessages []router.MessageContent
	var tokenMessages []token.MessageContent
	systemText := anthropicReq.SystemText()

	for _, msg := range anthropicReq.Messages {
		blocks := msg.ContentBlocks()
		content := extractTextFromBlocks(blocks)
		mc := router.MessageContent{
			Role:    msg.Role,
			Content: content,
		}
		routerMessages = append(routerMessages, mc)
		tokenMessages = append(tokenMessages, token.MessageContent{
			Role:    msg.Role,
			Content: content,
		})
	}

	// Count tokens.
	tokenCount, err := h.tokenCounter.CountMessages(systemText, tokenMessages)
	if err != nil {
		h.logger.Warn("failed to count tokens", "error", err)
		tokenCount = 0
	}

	// Route to appropriate model.
	routeResult, err := h.modelRouter.Route(routerMessages, tokenCount)
	if err != nil {
		h.sendError(w, http.StatusInternalServerError, "routing failed", err)
		return
	}

	h.logger.Info("routing request",
		"scenario", routeResult.Scenario,
		"model", routeResult.Primary.ModelID,
		"tokens", tokenCount,
	)

	// Build fallback chain.
	modelChain := routeResult.GetModelChain()

	if isStreaming {
		// Streaming: use ProxyStream for real-time SSE transformation
		h.handleStreaming(w, r, &anthropicReq, modelChain, rawBody)
	} else {
		// Non-streaming: execute with fallback and return full response
		h.handleNonStreaming(w, r, &anthropicReq, modelChain, rawBody)
	}
}

// handleStreaming handles a streaming request with real-time SSE proxying.
func (h *MessagesHandler) handleStreaming(
	w http.ResponseWriter,
	r *http.Request,
	anthropicReq *types.MessageRequest,
	modelChain []config.ModelConfig,
	rawBody json.RawMessage,
) {
	// Use background context with timeout - detached from HTTP request context but with timeout
	// The HTTP context gets canceled when the client disconnects or when
	// concurrent requests interfere, but streaming needs to continue independently.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	rw := &responseWriter{ResponseWriter: w}

	for _, model := range modelChain {
		h.logger.Info("attempting streaming model", "model", model.ModelID)

		// Check if this is an Anthropic-native model (MiniMax)
		if client.IsAnthropicModel(model.ModelID) {
			// For MiniMax models, send raw Anthropic request to Anthropic endpoint
			if err := h.handleAnthropicStreaming(ctx, rw, rawBody, model.ModelID); err != nil {
				h.logger.Warn("anthropic streaming failed", "model", model.ModelID, "error", err)
				continue
			}
			h.logger.Info("streaming completed", "model", model.ModelID)
			return
		}

		// For OpenAI-compatible models, transform and send to OpenAI endpoint
		openaiReq, err := h.requestTransformer.TransformRequest(anthropicReq, model)
		if err != nil {
			h.logger.Warn("request transform failed", "model", model.ModelID, "error", err)
			continue
		}

		// Get streaming body from upstream
		streamBody, err := h.client.GetStreamingBody(ctx, model.ModelID, openaiReq)
		if err != nil {
			h.logger.Warn("streaming request failed", "model", model.ModelID, "error", err)
			continue
		}
		defer streamBody.Close()

		// Proxy the stream: transform OpenAI SSE → Anthropic SSE in real-time
		if err := h.streamHandler.ProxyStream(rw, streamBody, model.ModelID); err != nil {
			h.logger.Warn("stream proxy failed", "model", model.ModelID, "error", err)
			continue
		}

		h.logger.Info("streaming completed", "model", model.ModelID)
		return
	}

	// All models failed
	if !rw.wroteHeader {
		h.sendError(w, http.StatusBadGateway, "all streaming models failed", nil)
	} else {
		// Headers already sent - send error as SSE event
		h.sendStreamError(rw, "all upstream models failed")
	}
}

// handleAnthropicStreaming sends a raw Anthropic request to the Anthropic endpoint.
func (h *MessagesHandler) handleAnthropicStreaming(
	ctx context.Context,
	w http.ResponseWriter,
	rawBody json.RawMessage,
	modelID string,
) error {
	// Send raw Anthropic request to Anthropic endpoint
	resp, err := h.client.SendAnthropicRequest(ctx, rawBody, true)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Copy the response directly (already in Anthropic format)
	if _, err := io.Copy(w, resp.Body); err != nil {
		return fmt.Errorf("failed to copy response: %w", err)
	}

	return nil
}

// sendStreamError sends an error event in the SSE stream.
// Use this when headers have already been written.
func (h *MessagesHandler) sendStreamError(w http.ResponseWriter, message string) {
	h.logger.Error("sending stream error", "message", message)

	errorEvent := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    "api_error",
			"message": message,
		},
	}

	data, _ := json.Marshal(errorEvent)
	fmt.Fprintf(w, "event: error\ndata: %s\n\n", string(data))

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// handleNonStreaming handles a non-streaming request with fallback.
func (h *MessagesHandler) handleNonStreaming(
	w http.ResponseWriter,
	r *http.Request,
	anthropicReq *types.MessageRequest,
	modelChain []config.ModelConfig,
	rawBody json.RawMessage,
) {
	ctx := r.Context()

	result, responseBody, err := h.fallbackHandler.ExecuteWithFallback(
		ctx,
		modelChain,
		func(ctx context.Context, model config.ModelConfig) ([]byte, error) {
			// Check if this is an Anthropic-native model (MiniMax)
			if client.IsAnthropicModel(model.ModelID) {
				return h.executeAnthropicRequest(ctx, rawBody, model)
			}
			// Otherwise use OpenAI transformation
			return h.executeOpenAIRequest(ctx, anthropicReq, model)
		},
	)

	if err != nil {
		h.sendError(w, http.StatusBadGateway, "all models failed", err)
		return
	}

	h.logger.Info("request completed",
		"model", result.ModelID,
		"attempts", result.Attempted,
	)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(responseBody)
}

// executeAnthropicRequest executes a request to the Anthropic endpoint (for MiniMax models).
func (h *MessagesHandler) executeAnthropicRequest(
	ctx context.Context,
	rawBody json.RawMessage,
	model config.ModelConfig,
) ([]byte, error) {
	// Send raw Anthropic request to Anthropic endpoint
	resp, err := h.client.SendAnthropicRequest(ctx, rawBody, false)
	if err != nil {
		return nil, fmt.Errorf("anthropic request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read the response (already in Anthropic format)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	h.logger.Debug("anthropic response", "body", string(body))

	return body, nil
}

// executeOpenAIRequest executes a request to the OpenAI endpoint with transformation.
func (h *MessagesHandler) executeOpenAIRequest(
	ctx context.Context,
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
) ([]byte, error) {
	// Transform request to OpenAI format.
	openaiReq, err := h.requestTransformer.TransformRequest(anthropicReq, model)
	if err != nil {
		return nil, fmt.Errorf("request transform failed: %w", err)
	}

	// Log the transformed request for debugging
	reqJSON, _ := json.Marshal(openaiReq)
	h.logger.Debug("transformed OpenAI request", "body", string(reqJSON))

	// Handle non-streaming.
	resp, err := h.client.ChatCompletionNonStreaming(ctx, model.ModelID, openaiReq)
	if err != nil {
		return nil, fmt.Errorf("chat completion failed: %w", err)
	}

	// Log the raw response for debugging
	respJSON, _ := json.Marshal(resp)
	h.logger.Debug("OpenAI response", "body", string(respJSON))

	// Transform response to Anthropic format.
	anthropicResp, err := h.responseTransformer.TransformResponse(resp, model.ModelID)
	if err != nil {
		return nil, fmt.Errorf("response transform failed: %w", err)
	}

	return json.Marshal(anthropicResp)
}

// extractTextFromBlocks extracts plain text from Anthropic content blocks.
func extractTextFromBlocks(blocks []types.ContentBlock) string {
	var content string
	for _, block := range blocks {
		switch block.Type {
		case "text":
			content += block.Text
		case "tool_use":
			content += fmt.Sprintf("[Tool Use: %s]", block.Name)
		case "tool_result":
			content += block.TextContent()
		case "thinking":
			// Skip thinking blocks for text extraction
		case "image":
			content += "[Image]"
		}
	}
	return content
}

// sendError sends an error response in Anthropic format.
// Safe to call multiple times - subsequent calls are no-ops.
func (h *MessagesHandler) sendError(w http.ResponseWriter, statusCode int, message string, err error) {
	h.logger.Error("request error",
		"status", statusCode,
		"message", message,
		"error", err,
	)

	// Use the wrapped writer if available to prevent duplicate WriteHeader calls
	if rw, ok := w.(*responseWriter); ok && rw.wroteHeader {
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errorResp := transformer.TransformErrorResponse(statusCode, message)
	json.NewEncoder(w).Encode(errorResp)
}
