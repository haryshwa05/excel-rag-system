"use client";

import { Bot, User, FileText, Table2, AlertCircle } from "lucide-react";

export interface Source {
  file_name: string;
  sheet_name: string;
  row_start: number;
  row_end: number;
  score: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
  isError?: boolean;
}

export default function MessageList({ messages }: { messages: Message[] }) {
  if (messages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-full p-10 text-center anim-fade-in">
        <div className="max-w-sm space-y-5">
          <div className="space-y-1.5">
            <h3 className="text-[15px] font-semibold" style={{ color: "var(--color-text)" }}>
              Ask anything about your data
            </h3>
            <p className="text-[13px] leading-relaxed" style={{ color: "var(--color-text-2)" }}>
              Type a question below. The AI answers using only what&apos;s in your document.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 justify-center">
            {["How many rows are there?", "What are the column names?", "Most common category?", "Show top 5 by value"].map((q) => (
              <span
                key={q}
                className="text-[11px] px-2.5 py-1 rounded-md"
                style={{
                  background: "var(--color-raised)",
                  border: "1px solid var(--color-border)",
                  color: "var(--color-text-2)",
                }}
              >
                {q}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-[720px] mx-auto w-full px-4 py-6 space-y-5">
      {messages.map((msg, i) => (
        <div key={msg.id} className="anim-fade-up" style={{ animationDelay: `${Math.min(i * 0.02, 0.1)}s` }}>
          {msg.role === "user" ? (
            <UserMessage content={msg.content} />
          ) : (
            <AssistantMessage
              content={msg.content}
              sources={msg.sources}
              isStreaming={msg.isStreaming}
              isError={msg.isError}
            />
          )}
        </div>
      ))}
    </div>
  );
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end gap-2.5 items-end anim-slide-right">
      <div
        className="max-w-[78%] rounded-2xl rounded-br-sm px-4 py-3"
        style={{
          background: "linear-gradient(135deg, #6366f1 0%, #5254d6 100%)",
          boxShadow: "0 4px 20px rgba(99,102,241,0.25)",
        }}
      >
        <p className="text-[13.5px] leading-relaxed text-white whitespace-pre-wrap">{content}</p>
      </div>
      <div
        className="w-7 h-7 rounded-full flex items-center justify-center shrink-0"
        style={{ background: "var(--color-elevated)", border: "1px solid var(--color-border-mid)" }}
      >
        <User size={12} style={{ color: "var(--color-text-2)" }} />
      </div>
    </div>
  );
}

function AssistantMessage({
  content,
  sources,
  isStreaming,
  isError,
}: {
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
  isError?: boolean;
}) {
  return (
    <div className="flex justify-start gap-2.5 items-end anim-slide-left">
      <div
        className="w-7 h-7 rounded-full flex items-center justify-center shrink-0"
        style={{
          background: isError ? "var(--color-danger-dim)" : "var(--color-accent-dim)",
          border: `1px solid ${isError ? "rgba(239,68,68,0.25)" : "rgba(99,102,241,0.25)"}`,
        }}
      >
        {isError
          ? <AlertCircle size={12} style={{ color: "var(--color-danger)" }} />
          : <Bot size={12} style={{ color: "var(--color-accent)" }} />
        }
      </div>

      <div className="max-w-[80%] space-y-2.5">
        <div
          className="rounded-2xl rounded-tl-sm px-4 py-3"
          style={{
            background: isError ? "rgba(239,68,68,0.06)" : "var(--color-raised)",
            border: `1px solid ${isError ? "rgba(239,68,68,0.2)" : "var(--color-border)"}`,
          }}
        >
          {content ? (
            <div className="prose-msg">
              <FormattedMessage text={content} />
              {isStreaming && (
                <span
                  className="inline-block w-0.5 h-[1.1em] ml-0.5 cursor-blink align-text-bottom"
                  style={{ background: "var(--color-accent)", borderRadius: "1px" }}
                />
              )}
            </div>
          ) : isStreaming ? (
            <span className="dot-loader" style={{ color: "var(--color-text-3)" }}>
              <span /><span /><span />
            </span>
          ) : null}
        </div>

        {sources && sources.length > 0 && !isStreaming && (
          <div className="flex flex-wrap items-center gap-1.5 pl-1">
            <span className="text-[10px] font-medium uppercase tracking-wide" style={{ color: "var(--color-text-3)" }}>
              Sources
            </span>
            {sources.slice(0, 5).map((src, i) => {
              const isSummary = src.sheet_name === "summary";
              const label = isSummary
                ? "Summary"
                : src.sheet_name !== "default"
                  ? src.sheet_name
                  : `Rows ${src.row_start}–${src.row_end}`;
              return (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md font-medium"
                  style={{
                    background: isSummary ? "var(--color-accent-dim)" : "var(--color-elevated)",
                    border: `1px solid ${isSummary ? "rgba(99,102,241,0.2)" : "var(--color-border)"}`,
                    color: isSummary ? "var(--color-accent-text)" : "var(--color-text-2)",
                  }}
                >
                  {isSummary ? <Table2 size={9} /> : <FileText size={9} />}
                  {label}
                </span>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Message text formatter ─────────────────────────────────────── */

function FormattedMessage({ text }: { text: string }) {
  const segments = splitCodeBlocks(text);
  return (
    <>
      {segments.map((seg, i) =>
        seg.type === "code-block" ? (
          <pre key={i}><code>{seg.content}</code></pre>
        ) : (
          <InlineContent key={i} text={seg.content} />
        )
      )}
    </>
  );
}

type Seg = { type: "text" | "code-block"; content: string };

function splitCodeBlocks(text: string): Seg[] {
  const out: Seg[] = [];
  const re = /```(?:[^\n]*)?\n?([\s\S]*?)```/g;
  let last = 0, m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ type: "text", content: text.slice(last, m.index) });
    out.push({ type: "code-block", content: m[1].trim() });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ type: "text", content: text.slice(last) });
  return out;
}

function InlineContent({ text }: { text: string }) {
  const paragraphs = text.split(/\n{2,}/);
  return (
    <>
      {paragraphs.map((para, pi) => {
        const trimmed = para.trim();
        if (!trimmed) return null;

        if (/^#{2,3}\s/.test(trimmed))
          return <h3 key={pi}>{trimmed.replace(/^#{2,3}\s/, "")}</h3>;

        const lines = trimmed.split("\n");
        const isList = lines.some((l) => /^[-*•]\s/.test(l.trim()));
        if (isList) return (
          <ul key={pi}>
            {lines.filter((l) => /^[-*•]\s/.test(l.trim())).map((l, li) => (
              <li key={li}><InlineSpans text={l.replace(/^[-*•]\s/, "")} /></li>
            ))}
          </ul>
        );

        const isNumbered = lines.some((l) => /^\d+\.\s/.test(l.trim()));
        if (isNumbered) return (
          <ol key={pi}>
            {lines.filter((l) => /^\d+\.\s/.test(l.trim())).map((l, li) => (
              <li key={li}><InlineSpans text={l.replace(/^\d+\.\s/, "")} /></li>
            ))}
          </ol>
        );

        return (
          <p key={pi}>
            {lines.map((line, li) => (
              <span key={li}>
                <InlineSpans text={line} />
                {li < lines.length - 1 && "\n"}
              </span>
            ))}
          </p>
        );
      })}
    </>
  );
}

function InlineSpans({ text }: { text: string }) {
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  const parts: React.ReactNode[] = [];
  let last = 0, m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) parts.push(<strong key={m.index}>{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith("`")) parts.push(<code key={m.index}>{tok.slice(1, -1)}</code>);
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}
