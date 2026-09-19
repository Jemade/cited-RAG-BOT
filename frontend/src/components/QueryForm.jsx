import React, { useState } from 'react';
import { Search, Sliders, ChevronDown, ChevronUp, Sparkles, HelpCircle } from 'lucide-react';

export default function QueryForm({
  query,
  setQuery,
  retrievalMode,
  setRetrievalMode,
  confidenceThreshold,
  setConfidenceThreshold,
  onSubmit,
  isLoading,
  disabled
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const sampleQueries = [
    "What optimizer was used for training and what were its hyperparameters?",
    "What BLEU score did the big Transformer model achieve on English-to-German?",
    "How many warmup steps were used in the learning rate schedule?",
    "What was the total training cost of GPT-4 in dollar terms?"
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading && !disabled) {
      onSubmit();
    }
  };

  return (
    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
            Ask a Document Question
          </label>
          <div className="relative">
            <textarea
              rows={3}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. What learning rate warmup schedule was used during training?"
              disabled={disabled || isLoading}
              className="w-full text-sm p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none resize-none"
            />
            <button
              type="submit"
              disabled={disabled || isLoading || !query.trim()}
              className="absolute bottom-3 right-3 px-4 py-1.5 bg-brand-600 hover:bg-brand-700 disabled:bg-slate-300 text-white rounded-md text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm"
            >
              {isLoading ? (
                <span>Searching...</span>
              ) : (
                <>
                  <Search className="w-3.5 h-3.5" />
                  <span>Ask CiteRAG</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Suggestion Chips */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[11px] text-slate-400 font-medium">Try:</span>
          {sampleQueries.map((sq, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setQuery(sq)}
              className="text-[11px] bg-slate-100 hover:bg-slate-200 text-slate-600 px-2 py-0.5 rounded-full transition-colors truncate max-w-[280px]"
              title={sq}
            >
              {sq}
            </button>
          ))}
        </div>

        {/* Retrieval Mode Selector */}
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1.5">
            Retrieval Strategy
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[
              { id: 'hybrid_rerank', label: 'Hybrid + Rerank', desc: 'RRF + Cross-Encoder (Recommended)' },
              { id: 'hybrid', label: 'Hybrid (RRF)', desc: 'Dense + BM25 Fusion' },
              { id: 'vector', label: 'Vector Only', desc: 'Dense pgvector similarity' },
              { id: 'bm25', label: 'BM25 Only', desc: 'Keyword term matching' },
            ].map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setRetrievalMode(m.id)}
                className={`p-2 rounded-lg border text-left text-xs transition-all ${
                  retrievalMode === m.id
                    ? 'border-brand-500 bg-brand-50 text-brand-900 font-medium ring-1 ring-brand-500'
                    : 'border-slate-200 hover:border-slate-300 text-slate-700 bg-white'
                }`}
              >
                <div className="font-semibold text-[12px]">{m.label}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">{m.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Settings Toggle */}
        <div className="pt-1">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 font-medium"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Advanced Parameters</span>
            {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {showAdvanced && (
            <div className="mt-3 p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-3">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium text-slate-700 flex items-center gap-1">
                    Confidence Refusal Threshold (Cross-Encoder Logit)
                  </span>
                  <span className="font-mono font-semibold text-brand-700">{confidenceThreshold}</span>
                </div>
                <input
                  type="range"
                  min="-6.0"
                  max="4.0"
                  step="0.5"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full accent-brand-600"
                />
                <p className="text-[10px] text-slate-500 mt-1">
                  If top chunk relevance score is below this threshold, CiteRAG will refuse to answer instead of hallucinating.
                </p>
              </div>
            </div>
          )}
        </div>
      </form>
    </div>
  );
}
