/**
 * hybrid-llm.ts - Compositor that routes LLM operations between remote and local backends
 *
 * Embed/rerank → remote (GPU-heavy, benefits from offloading)
 * Generate → local LlamaCpp
 * expandQuery → remote when configured, otherwise local
 * tokenize/countTokens → local LlamaCpp (CPU-cheap, needed for chunking)
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  LLMExpandQueryOptions,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";
import type { Token as LlamaToken } from "node-llama-cpp";
import { RemoteLLM } from "./remote-llm.js";

export class HybridLLM implements LLM {
  constructor(
    private readonly remote: LLM,
    private readonly local: LLM,
  ) {}

  get embedModelName(): string {
    return this.remote.embedModelName;
  }

  get generateModelName(): string {
    return this.local.generateModelName;
  }

  get expandModelName(): string {
    if (this.remote instanceof RemoteLLM && this.remote.supportsExpand) {
      return this.remote.expandModelName ?? this.remote.generateModelName;
    }
    return this.local.expandModelName ?? this.local.generateModelName;
  }

  get rerankModelName(): string {
    if (this.remote instanceof RemoteLLM && !this.remote.supportsRerank) {
      return this.local.rerankModelName;
    }
    return this.remote.rerankModelName;
  }

  get usesRemoteEmbedding(): boolean {
    return this.remote.usesRemoteEmbedding === true;
  }

  // Route to remote
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.remote.embed(text, options);
  }

  embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    return this.remote.embedBatch(texts, options);
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    // Remote configured: never silently fall back to a local GGUF reranker — it
    // would pull a multi-GB model onto remote-only hosts (same failure mode as
    // expandQuery). Without a remote rerank model, or on remote error, degrade
    // to an identity rerank that keeps candidates in their incoming order.
    if (this.remote instanceof RemoteLLM) {
      if (!this.remote.supportsRerank) {
        return this.identityRerank(documents);
      }
      try {
        return await this.remote.rerank(query, documents, options);
      } catch (error) {
        console.error("Remote rerank failed; keeping candidate order (no local fallback):", error);
        return this.identityRerank(documents);
      }
    }
    return this.local.rerank(query, documents, options);
  }

  /**
   * Identity rerank: preserve the incoming candidate order without scoring via
   * any model. Scores descend within [0.5, 1.0] so ordering is stable and no
   * candidate is dropped by a downstream min-score filter. Used when remote
   * reranking is configured-but-modelless or errors, so we never auto-download
   * a local GGUF reranker.
   */
  private identityRerank(documents: RerankDocument[]): RerankResult {
    const span = Math.max(1, documents.length - 1);
    return {
      model: "none",
      results: documents.map((d, i) => ({
        file: d.file,
        score: 1 - 0.5 * (i / span),
        index: i,
      })),
    };
  }

  // Route to local
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this.local.generate(prompt, options);
  }

  tokenize(text: string): Promise<readonly LlamaToken[]> {
    return this.local.tokenize(text);
  }

  detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    return this.local.detokenize(tokens);
  }

  async expandQuery(query: string, options?: LLMExpandQueryOptions): Promise<Queryable[]> {
    // Route to remote whenever a remote LLM is configured — even without a
    // dedicated expand model. RemoteLLM degrades to a deterministic raw-query
    // passthrough (lex/vec/hyde = the query itself) when no expand model is set
    // or the expand call fails, so we never silently pull a multi-GB local GGUF
    // onto remote-only hosts. Local expansion is used only when there is no
    // remote LLM configured at all.
    if (this.remote instanceof RemoteLLM) {
      try {
        return await this.remote.expandQuery(query, options);
      } catch (error) {
        console.error("Remote query expansion failed; falling back to local expansion:", error);
      }
    }
    return this.local.expandQuery(query, options);
  }

  modelExists(model: string): Promise<ModelInfo> {
    return this.local.modelExists(model);
  }

  async dispose(): Promise<void> {
    await Promise.all([this.remote.dispose(), this.local.dispose()]);
  }
}
