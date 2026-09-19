import React from 'react';
import { Layers, Bookmark, CheckCircle2 } from 'lucide-react';

export default function RetrievalInspect({ retrievedChunks, citations, onInspectChunk }) {
  if (!retrievedChunks || retrievedChunks.length === 0) return null;

  return (
    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
          <Layers className="w-4 h-4 text-brand-600" />
          Retrieved Chunks & Ranking Inspection ({retrievedChunks.length})
        </h3>
        <span className="text-[11px] text-slate-400">
          Ranked by Cross-Encoder / Fusion
        </span>
      </div>

      <div className="space-y-2.5">
        {retrievedChunks.map((chunk, idx) => {
          const isCited = citations?.some(c => c.chunk_id === chunk.chunk_id);
          const meta = chunk.metadata || {};

          return (
            <div
              key={chunk.chunk_id || idx}
              onClick={() => onInspectChunk({
                documentId: chunk.document_id,
                page: chunk.page_number,
                chunkId: chunk.chunk_id,
                snippet: chunk.text.slice(0, 180)
              })}
              className={`p-3 rounded-lg border text-xs cursor-pointer transition-all hover:border-brand-400 ${
                isCited
                  ? 'border-brand-300 bg-brand-50/40'
                  : 'border-slate-200 bg-slate-50/50 hover:bg-white'
              }`}
            >
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-slate-700 bg-slate-200 px-1.5 py-0.5 rounded text-[10px]">
                    #{chunk.rank}
                  </span>
                  <span className="font-semibold text-brand-800 flex items-center gap-1">
                    <Bookmark className="w-3 h-3" /> Page {chunk.page_number}
                  </span>
                  {chunk.section && (
                    <span className="text-slate-500 text-[11px] truncate max-w-[200px]">
                      {chunk.section}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 text-[10px] font-mono">
                  {meta.rerank_score !== undefined && (
                    <span className="bg-purple-50 text-purple-700 px-1.5 py-0.5 rounded border border-purple-200" title="Cross-Encoder Logit">
                      Rerank: {meta.rerank_score}
                    </span>
                  )}
                  {meta.rrf_score !== undefined && (
                    <span className="bg-sky-50 text-sky-700 px-1.5 py-0.5 rounded border border-sky-200" title="RRF Score">
                      RRF: {meta.rrf_score.toFixed(4)}
                    </span>
                  )}
                  {meta.dense_score !== undefined && (
                    <span className="bg-emerald-50 text-emerald-700 px-1.5 py-0.5 rounded border border-emerald-200" title="Dense Cosine Similarity">
                      Dense: {meta.dense_score !== null ? Number(meta.dense_score).toFixed(3) : '-'}
                    </span>
                  )}
                  {meta.bm25_score !== undefined && (
                    <span className="bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded border border-amber-200" title="BM25 Score">
                      BM25: {meta.bm25_score !== null ? Number(meta.bm25_score).toFixed(2) : '-'}
                    </span>
                  )}

                  {isCited && (
                    <span className="flex items-center gap-0.5 text-emerald-700 font-semibold bg-emerald-100 px-1.5 py-0.5 rounded">
                      <CheckCircle2 className="w-3 h-3" /> Cited
                    </span>
                  )}
                </div>
              </div>

              <p className="text-slate-600 line-clamp-3 text-[11px] leading-relaxed">
                {chunk.text}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
