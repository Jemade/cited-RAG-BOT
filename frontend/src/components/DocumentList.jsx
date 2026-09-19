import React from 'react';
import { FileText, CheckCircle2, Clock, AlertTriangle, Trash2 } from 'lucide-react';

export default function DocumentList({ documents, selectedDocId, onSelectDoc, onDeleteDoc }) {
  if (!documents || documents.length === 0) {
    return (
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
        <p className="text-sm text-slate-400">No documents ingested yet.</p>
        <p className="text-xs text-slate-400 mt-1">Upload a PDF above to begin.</p>
      </div>
    );
  }

  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider">
          Indexed Documents ({documents.length})
        </h3>
        {selectedDocId && (
          <button
            onClick={() => onSelectDoc(null)}
            className="text-xs text-brand-600 hover:text-brand-800 font-medium"
          >
            Query All Documents
          </button>
        )}
      </div>

      <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
        {documents.map((doc) => {
          const isSelected = selectedDocId === doc.id;
          return (
            <div
              key={doc.id}
              onClick={() => onSelectDoc(doc.id)}
              className={`p-3 rounded-lg border text-left cursor-pointer transition-all flex items-start justify-between gap-2 ${
                isSelected
                  ? 'border-brand-500 bg-brand-50 shadow-sm'
                  : 'border-slate-200 hover:border-slate-300 bg-white'
              }`}
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <FileText className={`w-4 h-4 shrink-0 ${isSelected ? 'text-brand-600' : 'text-slate-400'}`} />
                  <p className="text-xs font-semibold text-slate-800 truncate" title={doc.filename}>
                    {doc.filename}
                  </p>
                </div>
                <div className="flex items-center gap-3 mt-1.5 text-[11px] text-slate-500">
                  <span>{doc.page_count} pages</span>
                  <span>•</span>
                  <span>{doc.chunk_count} chunks</span>
                  <span>•</span>
                  <span className="flex items-center gap-1">
                    {doc.status === 'indexed' && (
                      <span className="inline-flex items-center gap-0.5 text-emerald-600 font-medium">
                        <CheckCircle2 className="w-3 h-3" /> Ready
                      </span>
                    )}
                    {doc.status === 'processing' && (
                      <span className="inline-flex items-center gap-0.5 text-amber-600 font-medium">
                        <Clock className="w-3 h-3 animate-spin" /> Ingesting
                      </span>
                    )}
                    {doc.status === 'failed' && (
                      <span className="inline-flex items-center gap-0.5 text-rose-600 font-medium">
                        <AlertTriangle className="w-3 h-3" /> Failed
                      </span>
                    )}
                  </span>
                </div>
              </div>

              {onDeleteDoc && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteDoc(doc.id);
                  }}
                  className="p-1 text-slate-400 hover:text-rose-600 rounded transition-colors"
                  title="Delete document"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
