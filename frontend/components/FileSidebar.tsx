"use client";

import { FileSpreadsheet, Trash2, Plus, FolderOpen, Hash } from "lucide-react";
import { FileInfo, deleteFile as apiDeleteFile } from "@/lib/api";

interface FileSidebarProps {
  files: FileInfo[];
  activeFile: FileInfo | null;
  onSelectFile: (f: FileInfo) => void;
  onDeleteFile: (fileId: string) => void;
  onUploadClick: () => void;
}

export default function FileSidebar({
  files,
  activeFile,
  onSelectFile,
  onDeleteFile,
  onUploadClick,
}: FileSidebarProps) {
  const handleDelete = async (e: React.MouseEvent, fileId: string) => {
    e.stopPropagation();
    try {
      await apiDeleteFile(fileId);
      onDeleteFile(fileId);
    } catch {
      // ignore
    }
  };

  return (
    <div className="h-full flex flex-col" style={{ background: "var(--color-surface-1)" }}>
      {/* Section label */}
      <div className="px-4 pt-5 pb-2">
        <p className="text-[10px] font-semibold uppercase tracking-[0.12em]"
           style={{ color: "var(--color-ink-faint)" }}>
          Files
        </p>
      </div>

      {/* File list */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {files.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-14 px-4 text-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                 style={{ background: "var(--color-surface-3)" }}>
              <FolderOpen size={16} style={{ color: "var(--color-ink-faint)" }} />
            </div>
            <p className="text-xs leading-relaxed" style={{ color: "var(--color-ink-faint)" }}>
              No files yet
            </p>
          </div>
        ) : (
          files.map((file) => {
            const active = activeFile?.file_id === file.file_id;
            return (
              <div
                key={file.file_id}
                role="button"
                tabIndex={0}
                onClick={() => onSelectFile(file)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelectFile(file);
                  }
                }}
                className="w-full text-left px-2.5 py-2 rounded-lg flex items-center gap-2.5 cursor-pointer group relative transition-all duration-150 outline-none focus-visible:ring-1"
                style={{
                  background: active ? "var(--color-accent-soft)" : "transparent",
                  boxShadow: active ? "inset 0 0 0 1px var(--color-border-strong)" : "none",
                  // focus ring color
                }}
              >
                {/* Active indicator bar */}
                {active && (
                  <span
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-full"
                    style={{ background: "var(--color-accent)" }}
                  />
                )}

                <div
                  className="w-7 h-7 rounded-md flex items-center justify-center shrink-0 transition-all"
                  style={{
                    background: active ? "var(--color-accent-soft)" : "var(--color-surface-3)",
                  }}
                >
                  <FileSpreadsheet
                    size={13}
                    style={{ color: active ? "var(--color-accent)" : "var(--color-ink-muted)" }}
                  />
                </div>

                <div className="flex-1 min-w-0">
                  <p
                    className="text-[12.5px] font-medium truncate leading-tight"
                    style={{ color: active ? "var(--color-ink)" : "var(--color-ink-muted)" }}
                  >
                    {file.file_name}
                  </p>
                  {file.chunks != null && (
                    <p className="flex items-center gap-1 text-[10px] mt-0.5"
                       style={{ color: "var(--color-ink-faint)" }}>
                      <Hash size={9} />
                      {file.chunks.toLocaleString()} chunks
                    </p>
                  )}
                </div>

                <button
                  onClick={(e) => handleDelete(e, file.file_id)}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded-md transition-all shrink-0 focus:outline-none"
                  style={{ color: "var(--color-ink-faint)" }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.color = "var(--color-danger)";
                    (e.currentTarget as HTMLElement).style.background = "var(--color-danger-soft)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.color = "var(--color-ink-faint)";
                    (e.currentTarget as HTMLElement).style.background = "transparent";
                  }}
                  title="Delete file"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* Upload button */}
      <div className="p-3" style={{ borderTop: "1px solid var(--color-border)" }}>
        <button
          onClick={onUploadClick}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-[12.5px] font-medium transition-all duration-150 active:scale-[0.98]"
          style={{
            background: "var(--color-accent-soft)",
            color: "var(--color-accent-text)",
            border: "1px solid var(--color-border-strong)",
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.background = "var(--color-accent)";
            (e.currentTarget as HTMLElement).style.color = "#fff";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.background = "var(--color-accent-soft)";
            (e.currentTarget as HTMLElement).style.color = "var(--color-accent-text)";
          }}
        >
          <Plus size={14} />
          New file
        </button>
      </div>
    </div>
  );
}
