import React, { useState, useEffect } from 'react';
import { X, ChevronLeft, ChevronRight, FileText, Image as ImageIcon, Loader2 } from 'lucide-react';

export default function SourceViewer({ inspectData, onClose }) {
  if (!inspectData) return null;

  const { documentId, page: initialPage, snippet, chunkId } = inspectData;
  const [currentPage, setCurrentPage] = useState(initialPage || 1);
  const [pageData, setPageData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState('both'); // 'both', 'image', 'text'

  useEffect(() => {
    setCurrentPage(initialPage);
  }, [initialPage]);

  useEffect(() => {
    if (!documentId || !currentPage) return;

    let isMounted = true;
    setIsLoading(true);

    fetch(`/v1/documents/${documentId}/sources/${currentPage}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch page source');
        return res.json();
      })
      .then(data => {
        if (isMounted) {
          setPageData(data);
          setIsLoading(false);
        }
      })
      .catch(err => {
        if (isMounted) {
          console.error(err);
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [documentId, currentPage]);

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl border border-slate-200 w-full max-w-5xl h-[88vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-brand-100 rounded-lg text-brand-700">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-800">
                Source Document Inspector
              </h3>
              <p className="text-xs text-slate-500 font-mono">
                Doc ID: {documentId?.slice(0, 8)}... | Page {currentPage}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Page Navigation */}
            <div className="flex items-center gap-1 bg-white border border-slate-200 rounded-lg px-2 py-1 text-xs">
              <button
                disabled={currentPage <= 1}
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                className="p-1 text-slate-500 hover:text-slate-800 disabled:opacity-30"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="font-semibold text-slate-700 px-2 font-mono">Page {currentPage}</span>
              <button
                onClick={() => setCurrentPage(p => p + 1)}
                className="p-1 text-slate-500 hover:text-slate-800"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center bg-white border border-slate-200 rounded-lg p-0.5 text-xs">
              <button
                onClick={() => setViewMode('both')}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  viewMode === 'both' ? 'bg-brand-600 text-white' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Split
              </button>
              <button
                onClick={() => setViewMode('image')}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  viewMode === 'image' ? 'bg-brand-600 text-white' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Page Image
              </button>
              <button
                onClick={() => setViewMode('text')}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  viewMode === 'text' ? 'bg-brand-600 text-white' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Extracted Text
              </button>
            </div>

            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-200 rounded-lg transition-colors ml-2"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-hidden p-6 bg-slate-100 flex gap-6">
          {isLoading ? (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-400 gap-2">
              <Loader2 className="w-8 h-8 animate-spin text-brand-600" />
              <p className="text-xs">Loading page source & preview...</p>
            </div>
          ) : (
            <>
              {/* Left Column: PDF Page Image Preview */}
              {(viewMode === 'both' || viewMode === 'image') && (
                <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
                  <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 text-xs font-semibold text-slate-600 flex items-center gap-1.5">
                    <ImageIcon className="w-3.5 h-3.5 text-brand-600" />
                    <span>Visual Page Render (PyMuPDF Pixmap)</span>
                  </div>
                  <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-slate-200/50">
                    {pageData?.has_preview ? (
                      <img
                        src={`/v1/documents/${documentId}/preview/${currentPage}`}
                        alt={`Page ${currentPage}`}
                        className="max-h-full max-w-full object-contain rounded shadow-md border border-slate-300"
                      />
                    ) : (
                      <div className="text-xs text-slate-400">Page preview image not available</div>
                    )}
                  </div>
                </div>
              )}

              {/* Right Column: Chunk Text & Snippet Highlight */}
              {(viewMode === 'both' || viewMode === 'text') && (
                <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
                  <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 text-xs font-semibold text-slate-600 flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5 text-brand-600" />
                    <span>Supporting Chunks & Snippets</span>
                  </div>
                  <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {/* Highlighted Cited Snippet */}
                    {snippet && (
                      <div className="p-3 bg-amber-50 border-l-4 border-amber-500 rounded-r-lg text-xs space-y-1 shadow-xs">
                        <span className="font-semibold text-amber-900 block">Cited Snippet Match:</span>
                        <p className="text-amber-800 italic">"{snippet}"</p>
                      </div>
                    )}

                    {/* All chunks on this page */}
                    <div className="space-y-3">
                      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        Chunks Extracted on Page {currentPage} ({pageData?.chunks?.length || 0})
                      </h4>
                      {pageData?.chunks?.map((chunk, idx) => {
                        const isMatch = chunkId === chunk.id;
                        return (
                          <div
                            key={chunk.id}
                            className={`p-3 rounded-lg border text-xs leading-relaxed ${
                              isMatch
                                ? 'border-brand-500 bg-brand-50/60 ring-1 ring-brand-500'
                                : 'border-slate-200 bg-slate-50 text-slate-700'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1 text-[10px] text-slate-400 font-mono">
                              <span>Chunk #{chunk.chunk_index} ({chunk.id})</span>
                              {chunk.section && (
                                <span className="text-brand-700 font-medium truncate max-w-[200px]">
                                  {chunk.section}
                                </span>
                              )}
                            </div>
                            <p className="whitespace-pre-wrap">{chunk.text}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
