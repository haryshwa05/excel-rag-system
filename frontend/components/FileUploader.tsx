"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, FileSpreadsheet, FileText, CheckCircle, AlertCircle, Loader2, X } from "lucide-react";
import { uploadFile, UploadProgressEvent, UploadResult } from "@/lib/api";

interface Props {
  onUploadComplete: (result: UploadResult) => void;
}

type Stage = "idle" | "uploading" | "parsing" | "embedding" | "indexing" | "done" | "error";

interface Progress {
  stage: Stage;
  percent: number;
  message: string;
}

const STAGE_LABELS: Record<Stage, string> = {
  idle:      "Ready",
  uploading: "Uploading…",
  parsing:   "Parsing file…",
  embedding: "Generating embeddings…",
  indexing:  "Building search index…",
  done:      "Complete",
  error:     "Error",
};

const ALLOWED = [".csv", ".xlsx", ".xls", ".pdf"];
const MAX_MB  = 100;

export default function FileUploader({ onUploadComplete }: Props) {
  const [dragging,  setDragging]  = useState(false);
  const [file,      setFile]      = useState<File | null>(null);
  const [progress,  setProgress]  = useState<Progress | null>(null);
  const [result,    setResult]    = useState<UploadResult | null>(null);
  const [error,     setError]     = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const reset = () => { setFile(null); setProgress(null); setResult(null); setError(null); };

  const validateFile = (f: File): string | null => {
    const ext = "." + (f.name.split(".").pop()?.toLowerCase() ?? "");
    if (!ALLOWED.includes(ext)) return `Unsupported type "${ext}". Allowed: ${ALLOWED.join(", ")}`;
    if (f.size > MAX_MB * 1024 * 1024) return `File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Max: ${MAX_MB} MB.`;
    return null;
  };

  const handleFile = useCallback(async (f: File) => {
    const err = validateFile(f);
    if (err) { setError(err); return; }

    setFile(f);
    setError(null);
    setResult(null);
    setProgress({ stage: "uploading", percent: 2, message: "Uploading…" });

    try {
      const res = await uploadFile(f, (evt: UploadProgressEvent) => {
        if (evt.stage === "error") {
          setProgress({ stage: "error", percent: 0, message: evt.message ?? "Unknown error" });
          setError(evt.message ?? "Unknown error");
          return;
        }
        const stageMap: Record<string, Stage> = {
          parsing: "parsing", embedding: "embedding",
          storing: "embedding", indexing: "indexing", complete: "done",
        };
        const mapped = stageMap[evt.stage] ?? "uploading";
        setProgress({ stage: mapped, percent: evt.percent ?? 0, message: evt.message ?? "" });
      });

      setProgress({ stage: "done", percent: 100, message: "Indexed and ready." });
      setResult(res);
      onUploadComplete(res);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed.";
      setProgress({ stage: "error", percent: 0, message: msg });
      setError(msg);
    }
  }, [onUploadComplete]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
    e.target.value = "";
  };

  const ext = file?.name.split(".").pop()?.toLowerCase();
  const FileIcon = ext === "pdf" ? FileText : FileSpreadsheet;
  const iconColor =
    ext === "pdf" ? "var(--color-pdf)" :
    ext === "csv" ? "var(--color-csv)" : "var(--color-excel)";

  /* ── Done state ────────────────────────────────────────────────── */
  if (result && progress?.stage === "done") {
    return (
      <div
        className="rounded-xl p-5 flex items-start gap-3 anim-fade-up"
        style={{ background: "var(--color-success-dim)", border: "1px solid rgba(16,185,129,0.25)" }}
      >
        <CheckCircle size={18} style={{ color: "var(--color-success)", flexShrink: 0 }} />
        <div className="flex-1 min-w-0">
          <p className="text-[13px] font-semibold" style={{ color: "var(--color-text)" }}>
            {result.file_name}
          </p>
          <p className="text-[12px] mt-0.5" style={{ color: "var(--color-text-2)" }}>
            {result.total_chunks} chunks indexed · Ready to query
          </p>
        </div>
        <button onClick={reset} className="p-1 rounded hover:bg-white/5">
          <X size={13} style={{ color: "var(--color-text-3)" }} />
        </button>
      </div>
    );
  }

  /* ── Processing state ──────────────────────────────────────────── */
  if (file && progress) {
    const pct     = Math.round(progress.percent);
    const isError = progress.stage === "error";

    return (
      <div
        className="rounded-xl p-5 space-y-4 anim-fade-up"
        style={{
          background: isError ? "var(--color-danger-dim)" : "var(--color-raised)",
          border: `1px solid ${isError ? "rgba(239,68,68,0.25)" : "var(--color-border-mid)"}`,
        }}
      >
        <div className="flex items-center gap-3">
          <FileIcon size={18} style={{ color: isError ? "var(--color-danger)" : iconColor, flexShrink: 0 }} />
          <div className="min-w-0 flex-1">
            <p className="text-[12.5px] font-medium truncate" style={{ color: "var(--color-text)" }}>{file.name}</p>
            <p className="text-[11px]" style={{ color: "var(--color-text-2)" }}>
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
          {isError ? (
            <button onClick={reset} className="p-1 rounded hover:bg-white/5">
              <X size={13} style={{ color: "var(--color-text-3)" }} />
            </button>
          ) : (
            <Loader2 size={14} className="anim-spin" style={{ color: "var(--color-accent)" }} />
          )}
        </div>

        {!isError && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px]" style={{ color: "var(--color-text-2)" }}>
                {STAGE_LABELS[progress.stage]}
              </span>
              <span className="text-[11px] font-medium" style={{ color: "var(--color-accent-text)" }}>
                {pct}%
              </span>
            </div>
            <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: "var(--color-elevated)" }}>
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${pct}%`, background: "linear-gradient(90deg, #6366f1, #a5b4fc)" }}
              />
            </div>
            <p className="text-[10.5px]" style={{ color: "var(--color-text-3)" }}>{progress.message}</p>
          </div>
        )}

        {isError && (
          <div className="flex items-center gap-2">
            <AlertCircle size={13} style={{ color: "var(--color-danger)" }} />
            <p className="text-[12px]" style={{ color: "var(--color-danger)" }}>{progress.message}</p>
          </div>
        )}
      </div>
    );
  }

  /* ── Drop zone ─────────────────────────────────────────────────── */
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className="relative cursor-pointer rounded-xl p-6 flex flex-col items-center gap-3 transition-all select-none"
      style={{
        background: dragging ? "var(--color-accent-dim)" : "var(--color-raised)",
        border: `1.5px dashed ${dragging ? "var(--color-accent)" : "var(--color-border-mid)"}`,
      }}
    >
      <input ref={inputRef} type="file" accept={ALLOWED.join(",")} onChange={onInputChange} className="sr-only" />

      <div
        className="w-11 h-11 rounded-xl flex items-center justify-center"
        style={{
          background: dragging ? "var(--color-accent-mid)" : "var(--color-elevated)",
          border: `1px solid ${dragging ? "rgba(99,102,241,0.3)" : "var(--color-border)"}`,
        }}
      >
        <Upload size={18} style={{ color: dragging ? "var(--color-accent)" : "var(--color-text-2)" }} />
      </div>

      <div className="text-center">
        <p className="text-[13px] font-medium" style={{ color: "var(--color-text)" }}>
          {dragging ? "Drop to upload" : "Drop file or click to browse"}
        </p>
        <p className="text-[11px] mt-0.5" style={{ color: "var(--color-text-3)" }}>
          CSV · XLSX · XLS · PDF — up to {MAX_MB} MB
        </p>
      </div>

      {error && (
        <div
          className="flex items-center gap-1.5 text-[12px] px-3 py-1.5 rounded-md"
          style={{ color: "var(--color-danger)", background: "var(--color-danger-dim)", border: "1px solid rgba(239,68,68,0.2)" }}
        >
          <AlertCircle size={12} />
          {error}
        </div>
      )}
    </div>
  );
}
