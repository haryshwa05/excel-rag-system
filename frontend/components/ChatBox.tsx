"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Square } from "lucide-react";
import { queryStream, FileInfo } from "@/lib/api";
import MessageList, { Message } from "./MessageList";

interface ChatBoxProps {
  activeFile: FileInfo;
}


let _msgId = 0;
const nextId = () => `msg-${++_msgId}`;

export default function ChatBox({ activeFile }: ChatBoxProps) {
  const [messages,    setMessages]    = useState<Message[]>([]);
  const [input,       setInput]       = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const scrollRef  = useRef<HTMLDivElement>(null);
  const abortRef   = useRef<(() => void) | null>(null);
  const prevFileId = useRef<string>("");

  // Reset chat when file changes
  useEffect(() => {
    if (prevFileId.current !== activeFile.file_id) {
      prevFileId.current = activeFile.file_id;
      setMessages([]);
    }
  }, [activeFile.file_id]);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const stopStreaming = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setIsStreaming(false);
    setMessages((prev) =>
      prev.map((m) => m.isStreaming ? { ...m, isStreaming: false } : m)
    );
  }, []);

  const handleSend = useCallback(async () => {
    const q = input.trim();
    if (!q || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    const userMsg: Message = { id: nextId(), role: "user", content: q };
    const assistantId = nextId();
    const assistantMsg: Message = { id: assistantId, role: "assistant", content: "", isStreaming: true };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    let accumulated = "";
    let aborted = false;

    const { abort } = queryStream(
      { question: q, file_id: activeFile.file_id, chat_history: [] },
      {
        onToken: (token) => {
          accumulated += token;
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, content: accumulated } : m)
          );
        },
        onSources: () => {},
        onDone: () => {
          if (aborted) return;
          setIsStreaming(false);
          abortRef.current = null;
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, content: accumulated, isStreaming: false } : m)
          );
        },
        onError: (err) => {
          if (aborted) return;
          setIsStreaming(false);
          abortRef.current = null;
          const errContent = accumulated
            ? `${accumulated}\n\n⚠ ${err}`
            : `Error: ${err}`;
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, content: errContent, isStreaming: false } : m)
          );
        },
      }
    );
    abortRef.current = () => { aborted = true; abort(); };
  }, [input, isStreaming, activeFile.file_id]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleStop = () => stopStreaming();

  const disabled = false;

  const extColors: Record<string, string> = {
    csv: "var(--color-csv)", xlsx: "var(--color-excel)",
    xls: "var(--color-excel)", pdf: "var(--color-pdf)",
  };
  const ext = activeFile.file_name.split(".").pop()?.toLowerCase() ?? "";
  const extColor = extColors[ext] ?? "var(--color-text-2)";

  return (
    <div className="flex flex-col h-full">
      {/* Context bar */}
      <div
        className="shrink-0 flex items-center justify-between px-5 py-2.5 border-b"
        style={{ background: "var(--color-surface)", borderColor: "var(--color-border)" }}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <span
            className="text-[9px] font-bold px-1.5 py-0.5 rounded shrink-0 uppercase"
            style={{
              color: extColor,
              background: `${extColor}20`,
              border: `1px solid ${extColor}40`,
              letterSpacing: "0.04em",
            }}
          >
            {ext}
          </span>
          <div className="min-w-0">
            <p className="text-[13px] font-medium truncate" style={{ color: "var(--color-text)" }}>
              {activeFile.file_name}
            </p>
            {activeFile.chunks != null && (
              <p className="text-[11px]" style={{ color: "var(--color-text-3)" }}>
                {activeFile.chunks.toLocaleString()} chunks indexed
              </p>
            )}
          </div>
        </div>
        {isStreaming && (
          <div className="flex items-center gap-1.5 text-[11px] anim-fade-in" style={{ color: "var(--color-accent-text)" }}>
            <span className="dot-loader"><span /><span /><span /></span>
            <span>Generating…</span>
          </div>
        )}
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto"
        style={{ background: "var(--color-bg)" }}
      >
        <MessageList messages={messages} />
      </div>

      {/* Input */}
      <div
        className="shrink-0 px-4 pb-5 pt-3"
        style={{ background: "var(--color-bg)", borderTop: "1px solid var(--color-border)" }}
      >
        <div
          className="max-w-[720px] mx-auto flex items-end gap-2.5 rounded-xl p-3"
          style={{
            background: "var(--color-raised)",
            border: "1px solid var(--color-border-mid)",
            boxShadow: "0 2px 12px rgba(0,0,0,0.2)",
          }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask anything about ${activeFile.file_name}…`}
            rows={1}
            className="flex-1 bg-transparent resize-none outline-none text-[13.5px] leading-relaxed min-h-[22px] max-h-[140px] overflow-y-auto"
            style={{ color: "var(--color-text)", fontFamily: "var(--font-sans)" }}
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
            }}
            disabled={isStreaming}
          />
          {isStreaming ? (
            <button
              onClick={handleStop}
              title="Stop"
              className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 hover:brightness-110 transition-all"
              style={{ background: "var(--color-danger-dim)", border: "1px solid rgba(239,68,68,0.25)" }}
            >
              <Square size={13} style={{ color: "var(--color-danger)" }} />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              title="Send (Enter)"
              className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-all disabled:opacity-30 disabled:cursor-not-allowed hover:brightness-110 active:scale-95"
              style={{ background: "var(--color-accent)" }}
            >
              <Send size={13} className="text-white" />
            </button>
          )}
        </div>
        <p className="text-center text-[10px] mt-2" style={{ color: "var(--color-text-3)" }}>
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
