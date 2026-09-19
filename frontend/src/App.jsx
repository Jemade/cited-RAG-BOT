import React, { useState, useEffect } from 'react';
import { BookOpen, ShieldCheck, Database, Layers, Sparkles, Activity } from 'lucide-react';
import DocumentUpload from './components/DocumentUpload';
import DocumentList from './components/DocumentList';
import QueryForm from './components/QueryForm';
import AnswerCard from './components/AnswerCard';
import SourceViewer from './components/SourceViewer';
import RetrievalInspect from './components/RetrievalInspect';

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [query, setQuery] = useState('');
  const [retrievalMode, setRetrievalMode] = useState('hybrid_rerank');
  const [confidenceThreshold, setConfidenceThreshold] = useState(-2.5);
  const [isLoading, setIsLoading] = useState(false);
  const [queryResult, setQueryResult] = useState(null);
  const [inspectData, setInspectData] = useState(null);
  const [metrics, setMetrics] = useState(null);

  // Load documents on mount
  const fetchDocuments = async () => {
    try {
      const resp = await fetch('/v1/documents');
      if (resp.ok) {
        const data = await resp.json();
        setDocuments(data);
        if (data.length > 0 && !selectedDocId) {
          setSelectedDocId(data[0].id);
        }
      }
    } catch (e) {
      console.error('Failed to load documents:', e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const resp = await fetch('/metrics');
      if (resp.ok) {
        const data = await resp.json();
        setMetrics(data);
      }
    } catch (e) {
      console.error('Failed to load metrics:', e);
    }
  };

  useEffect(() => {
    fetchDocuments();
    fetchMetrics();
  }, []);

  const handleUploadSuccess = (newDoc) => {
    fetchDocuments();
    fetchMetrics();
    setSelectedDocId(newDoc.id);
  };

  const handleDeleteDoc = async (docId) => {
    try {
      const resp = await fetch(`/v1/documents/${docId}`, { method: 'DELETE' });
      if (resp.ok) {
        if (selectedDocId === docId) setSelectedDocId(null);
        fetchDocuments();
        fetchMetrics();
      }
    } catch (e) {
      console.error('Failed to delete document:', e);
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setQueryResult(null);

    const endpoint = selectedDocId
      ? `/v1/documents/${selectedDocId}/query`
      : '/v1/query';

    try {
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          retrieval_mode: retrievalMode,
          confidence_threshold: confidenceThreshold,
          top_k: 20,
          top_n: 5
        })
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'Query failed' }));
        throw new Error(err.detail || 'Query request failed');
      }

      const data = await resp.json();
      setQueryResult(data);
      fetchMetrics();
    } catch (err) {
      alert(`Query error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100/70 text-slate-900 flex flex-col">
      {/* Top Navbar */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-brand-600 rounded-lg text-white shadow-sm">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
                CiteRAG
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-brand-50 text-brand-700 border border-brand-200">
                  Production RAG
                </span>
              </h1>
              <p className="text-[11px] text-slate-500">
                Precision Cited PDF Q&A with pgvector, BM25 & Cross-Encoder Reranking
              </p>
            </div>
          </div>

          {/* Quick Metrics */}
          {metrics && (
            <div className="hidden md:flex items-center gap-4 text-xs text-slate-500 font-mono">
              <span className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-slate-400" />
                <span>{metrics.total_documents} Docs</span>
              </span>
              <span>•</span>
              <span>{metrics.total_chunks} Chunks</span>
              <span>•</span>
              <span title="Total queries processed">
                {metrics.total_queries} Queries
              </span>
              <span>•</span>
              <span className="text-amber-600" title="Low-confidence refusal rate">
                {Math.round(metrics.refusal_rate * 100)}% Refusal Rate
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 flex-1 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Documents & Ingestion (4 cols) */}
          <div className="lg:col-span-4 space-y-5">
            <DocumentUpload onUploadSuccess={handleUploadSuccess} />
            <DocumentList
              documents={documents}
              selectedDocId={selectedDocId}
              onSelectDoc={setSelectedDocId}
              onDeleteDoc={handleDeleteDoc}
            />
          </div>

          {/* Right Column: Query & Answers & Inspection (8 cols) */}
          <div className="lg:col-span-8 space-y-5">
            <QueryForm
              query={query}
              setQuery={setQuery}
              retrievalMode={retrievalMode}
              setRetrievalMode={setRetrievalMode}
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              onSubmit={handleQuery}
              isLoading={isLoading}
              disabled={documents.length === 0}
            />

            {/* Answer Display */}
            {queryResult && (
              <AnswerCard
                result={queryResult}
                onCitationClick={(data) => {
                  setInspectData({
                    documentId: data.documentId || selectedDocId,
                    page: data.page,
                    chunkId: data.chunkId,
                    snippet: data.snippet
                  });
                }}
              />
            )}

            {/* Retrieval Score Inspection */}
            {queryResult && (
              <RetrievalInspect
                retrievedChunks={queryResult.retrieved_chunks}
                citations={queryResult.citations}
                onInspectChunk={(data) => setInspectData(data)}
              />
            )}
          </div>
        </div>
      </main>

      {/* Source Viewer Modal */}
      {inspectData && (
        <SourceViewer
          inspectData={inspectData}
          onClose={() => setInspectData(null)}
        />
      )}
    </div>
  );
}
