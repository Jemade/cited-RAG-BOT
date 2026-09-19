import React, { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

export default function DocumentUpload({ onUploadSuccess }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleUpload = async (file) => {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a valid PDF file.');
      return;
    }

    setError(null);
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch('/v1/documents', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errData.detail || 'Upload failed');
      }

      const doc = await resp.json();
      if (onUploadSuccess) {
        onUploadSuccess(doc);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-2">
        <FileText className="w-4 h-4 text-brand-600" /> Ingest PDF Document
      </h3>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${
          isDragging ? 'border-brand-500 bg-brand-50' : 'border-slate-300 hover:border-brand-400 bg-slate-50'
        } ${isUploading ? 'opacity-60 cursor-not-allowed' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
          disabled={isUploading}
        />

        {isUploading ? (
          <div className="flex flex-col items-center justify-center gap-2 text-slate-600">
            <Loader2 className="w-7 h-7 text-brand-600 animate-spin" />
            <p className="text-sm font-medium">Extracting pages & generating embeddings...</p>
            <p className="text-xs text-slate-400">Page-aware chunking in progress</p>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-1.5 text-slate-600">
            <Upload className="w-6 h-6 text-slate-400" />
            <p className="text-sm font-medium">Drop PDF here or click to browse</p>
            <p className="text-xs text-slate-400">Extracts pages, chunks, and creates pgvector embeddings</p>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-2.5 flex items-center gap-2 text-xs text-rose-600 bg-rose-50 p-2 rounded-md border border-rose-200">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
