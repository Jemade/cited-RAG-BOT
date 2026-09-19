import React from 'react';
import { ShieldCheck, AlertCircle, Bookmark, CheckCircle2, Clock, Zap } from 'lucide-react';

export default function AnswerCard({ result, onCitationClick }) {
  if (!result) return null;

  const { answer, citations, refused, refusal_reason, confidence_score, latency_ms, retrieval_mode, request_id } = result;

  // Helper to render text with clickable citation badges
  const renderFormattedAnswer = () => {
    if (!answer) return null;

    // Split text by citation pattern: [Page X]
    const parts = answer.split(/(\[Page\s*\d+\])/g);

    return parts.map((part, index) => {
      const match = part.match(/\[Page\s*(\d+)\]/i);
      if (match) {
        const pageNum = parseInt(match[1], 10);
        const matchedCitation = citations?.find(c => c.page_number === pageNum);

        return (
          <button
            key={index}
            onClick={() => onCitationClick({
              page: pageNum,
              chunkId: matchedCitation?.chunk_id,
              snippet: matchedCitation?.snippet,
              documentId: matchedCitation?.document_id
            })}
            className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded text-xs font-semibold bg-brand-100 hover:bg-brand-200 text-brand-800 border border-brand-300 transition-colors shadow-xs"
            title="Click to inspect verified source page and chunk snippet"
          >
            <Bookmark className="w-3 h-3 text-brand-600" />
            <span>Page {pageNum}</span>
          </button>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
      {/* Header status bar */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          {refused ? (
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full bg-amber-100 text-amber-800 border border-amber-200">
              <AlertCircle className="w-3.5 h-3.5 text-amber-600" /> Low-Confidence Refusal
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" /> Cited & Verified Answer
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-500 font-mono">
          {confidence_score !== null && (
            <span title="Top Cross-Encoder Score">
              Score: <strong className="text-slate-700">{confidence_score}</strong>
            </span>
          )}
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3 text-slate-400" /> {latency_ms} ms
          </span>
          <span className="capitalize bg-slate-100 px-2 py-0.5 rounded text-[11px] text-slate-600">
            {retrieval_mode}
          </span>
        </div>
      </div>

      {/* Answer Body */}
      {refused ? (
        <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg space-y-2">
          <p className="text-sm font-semibold text-amber-900">
            {answer}
          </p>
          {refusal_reason && (
            <p className="text-xs text-amber-700">
              <strong>System Notice:</strong> {refusal_reason}
            </p>
          )}
          <p className="text-xs text-amber-600">
            CiteRAG refuses to answer rather than hallucinating when retrieved document passages fall below the confidence threshold.
          </p>
        </div>
      ) : (
        <div className="text-sm text-slate-800 leading-relaxed font-normal">
          {renderFormattedAnswer()}
        </div>
      )}

      {/* Citations Summary Footer */}
      {!refused && citations && citations.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-100">
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> Supporting Document Citations ({citations.length})
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {citations.map((c, i) => (
              <div
                key={i}
                onClick={() => onCitationClick({
                  page: c.page_number,
                  chunkId: c.chunk_id,
                  snippet: c.snippet,
                  documentId: c.document_id
                })}
                className="p-2.5 rounded-lg border border-slate-200 hover:border-brand-300 hover:bg-brand-50/50 cursor-pointer transition-colors text-xs"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-brand-700 flex items-center gap-1">
                    <Bookmark className="w-3 h-3" /> Page {c.page_number}
                  </span>
                  <span className="text-[10px] text-emerald-600 font-mono bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                    Verified Match
                  </span>
                </div>
                <p className="text-slate-600 text-[11px] line-clamp-2 italic">
                  "{c.snippet}"
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
