"use client";

import { useState, useEffect, useCallback } from "react";
import { BarChart3, WifiOff, Loader2 } from "lucide-react";
import FileSidebar from "@/components/FileSidebar";
import ChatBox from "@/components/ChatBox";
import FileUploader from "@/components/FileUploader";
import { getFiles, FileInfo, UploadResult } from "@/lib/api";

type BackendStatus = "checking" | "online" | "offline";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000/api";

function useBackendStatus() {
  const [status, setStatus] = useState<BackendStatus>("checking");
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
        if (!cancelled) setStatus(res.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    };
    check();
    const id = setInterval(check, 15_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);
  return status;
}

export default function Home() {
  const [files,      setFiles]      = useState<FileInfo[]>([]);
  const [activeFile, setActiveFile] = useState<FileInfo | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [isLoading,  setIsLoading]  = useState(true);
  const backendStatus = useBackendStatus();

  useEffect(() => {
    let cancelled = false;
    getFiles().then((list) => {
      if (cancelled) return;
      setFiles(list);
      setIsLoading(false);
      if (list.length > 0) setActiveFile((cur) => cur ?? list[0]);
    });
    return () => { cancelled = true; };
  }, []);

  const handleUploadComplete = useCallback((result: UploadResult) => {
    const f: FileInfo = {
      file_id: result.file_id,
      file_name: result.file_name,
      chunks: result.total_chunks,
    };
    setFiles((prev) => [f, ...prev]);
    setActiveFile(f);
    setShowUpload(false);
  }, []);

  const handleDeleteFile = useCallback((fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.file_id !== fileId));
    setActiveFile((cur) => (cur?.file_id === fileId ? null : cur));
  }, []);

  return (
    <div className="h-screen flex overflow-hidden" style={{ background: "var(--color-bg)" }}>

      {/* ── Sidebar ──────────────────────────────────────────────── */}
      <aside
        className="w-[240px] shrink-0 hidden md:flex flex-col border-r"
        style={{ background: "var(--color-surface)", borderColor: "var(--color-border)" }}
      >
        {/* Logo header */}
        <div
          className="h-14 flex items-center gap-3 px-4 border-b shrink-0"
          style={{ borderColor: "var(--color-border)" }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: "var(--color-accent)" }}
          >
            <BarChart3 size={16} className="text-white" />
          </div>
          <div>
            <p className="text-[13px] font-semibold" style={{ color: "var(--color-text)" }}>
              DataRAG
            </p>
            <p className="text-[10px]" style={{ color: "var(--color-text-3)" }}>
              Document Intelligence
            </p>
          </div>
          <div className="ml-auto">
            {backendStatus === "checking" && (
              <Loader2 size={12} className="anim-spin" style={{ color: "var(--color-text-3)" }} />
            )}
            {backendStatus === "online" && (
              <span
                className="block w-2 h-2 rounded-full anim-pulse"
                style={{ background: "var(--color-success)" }}
              />
            )}
            {backendStatus === "offline" && (
              <span
                className="block w-2 h-2 rounded-full"
                style={{ background: "var(--color-danger)" }}
              />
            )}
          </div>
        </div>

        <FileSidebar
          files={files}
          activeFile={activeFile}
          onSelectFile={(f) => setActiveFile(f)}
          onDeleteFile={handleDeleteFile}
          onUploadClick={() => setShowUpload(true)}
        />
      </aside>

      {/* ── Main ─────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* Offline banner */}
        {backendStatus === "offline" && (
          <div
            className="flex items-center gap-2.5 px-4 py-2.5 text-xs shrink-0 anim-fade-in"
            style={{
              background: "var(--color-danger-dim)",
              borderBottom: "1px solid rgba(239,68,68,0.2)",
              color: "var(--color-danger)",
            }}
          >
            <WifiOff size={12} className="shrink-0" />
            <span>
              Backend offline — run{" "}
              <code
                className="px-1 py-0.5 rounded text-[11px]"
                style={{ background: "rgba(239,68,68,0.15)", fontFamily: "var(--font-mono)" }}
              >
                uvicorn main:app --reload
              </code>{" "}
              in the <code style={{ fontFamily: "var(--font-mono)" }}>backend/</code> folder.
            </span>
          </div>
        )}

        {isLoading ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3 anim-fade-in">
              <Loader2 size={24} className="anim-spin" style={{ color: "var(--color-accent)" }} />
              <p className="text-[13px]" style={{ color: "var(--color-text-2)" }}>Loading…</p>
            </div>
          </div>
        ) : !activeFile ? (
          <EmptyState onUpload={() => setShowUpload(true)} />
        ) : (
          <div className="flex-1 min-h-0">
            <ChatBox activeFile={activeFile} />
          </div>
        )}
      </div>

      {/* ── Upload modal ─────────────────────────────────────────── */}
      {showUpload && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 anim-fade-in"
          style={{ background: "rgba(0,0,0,0.65)", backdropFilter: "blur(6px)" }}
          onClick={() => setShowUpload(false)}
        >
          <div
            className="w-full max-w-[460px] rounded-2xl p-6 anim-fade-up"
            style={{
              background: "var(--color-raised)",
              border: "1px solid var(--color-border-mid)",
              boxShadow: "0 24px 60px rgba(0,0,0,0.55)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-5">
              <h2 className="text-[15px] font-semibold" style={{ color: "var(--color-text)" }}>
                Upload a document
              </h2>
              <p className="text-[12px] mt-1" style={{ color: "var(--color-text-2)" }}>
                Supports CSV, Excel (.xlsx / .xls), and PDF files up to 100 MB.
              </p>
            </div>
            <FileUploader onUploadComplete={handleUploadComplete} />
          </div>
        </div>
      )}
    </div>
  );
}

function EmptyState({ onUpload }: { onUpload: () => void }) {
  return (
    <div className="flex-1 flex items-center justify-center p-8 anim-fade-up">
      <div className="max-w-[380px] w-full text-center space-y-6">
        <div
          className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto"
          style={{
            background: "var(--color-accent-dim)",
            border: "1px solid rgba(99,102,241,0.25)",
          }}
        >
          <BarChart3 size={28} style={{ color: "var(--color-accent)" }} />
        </div>

        <div>
          <h2 className="text-[18px] font-semibold" style={{ color: "var(--color-text)" }}>
            No documents yet
          </h2>
          <p className="text-[13px] mt-1.5 leading-relaxed" style={{ color: "var(--color-text-2)" }}>
            Upload a spreadsheet or PDF to start asking questions about your data in plain English.
          </p>
        </div>

        <button
          onClick={onUpload}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-[13px] font-medium text-white transition-all hover:brightness-110 active:scale-[0.98]"
          style={{ background: "var(--color-accent)" }}
        >
          Upload a document
        </button>

        <div className="grid grid-cols-3 gap-3">
          {[
            { label: "CSV files",     color: "var(--color-csv)"   },
            { label: "Excel (.xlsx)", color: "var(--color-excel)" },
            { label: "PDF files",     color: "var(--color-pdf)"   },
          ].map((t) => (
            <div
              key={t.label}
              className="rounded-lg px-3 py-2 text-[11px] font-medium text-center"
              style={{
                background: "var(--color-raised)",
                border: "1px solid var(--color-border)",
                color: t.color,
              }}
            >
              {t.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
