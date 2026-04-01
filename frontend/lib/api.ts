// Direct backend URL avoids Next dev-server proxy stalls for large streaming uploads.
// Override via NEXT_PUBLIC_API_BASE if needed.
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000/api";

/* ── Types ─────────────────────────────────────────────────────── */

export interface FileInfo {
  file_id: string;
  file_name: string;
  chunks?: number;
  total_chunks?: number;
  total_rows?: number;
  sheets?: string[];
}

export interface UploadResult {
  file_id: string;
  file_name: string;
  total_chunks: number;
  status: string;
}

export interface UploadProgressEvent {
  stage: string;
  percent?: number;
  message?: string;
  file_id?: string;
  error?: string;
}

export interface QueryRequest {
  question: string;
  file_id?: string;
  sheet_name?: string;
  chat_history?: Array<{ role: string; content: string }>;
}

export interface Source {
  file_name: string;
  sheet_name: string;
  row_start: number;
  row_end: number;
  score: number;
}

export interface QueryStreamCallbacks {
  onToken: (token: string) => void;
  onSources?: (sources: Source[]) => void;
  onDone: () => void;
  onError: (err: string) => void;
}

/* ── File management ───────────────────────────────────────────── */

export async function getFiles(): Promise<FileInfo[]> {
  try {
    const res = await fetch(`${BASE}/files`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.files ?? []).map((f: FileInfo) => ({
      ...f,
      chunks: f.total_chunks ?? f.chunks ?? 0,
    }));
  } catch {
    return [];
  }
}

export async function deleteFile(fileId: string): Promise<void> {
  const res = await fetch(`${BASE}/files/${fileId}`, { method: "DELETE" });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail ?? "Delete failed.");
  }
}

/* ── File upload with SSE progress ─────────────────────────────── */

export async function uploadFile(
  file: File,
  onProgress: (e: UploadProgressEvent) => void
): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail ?? `Upload failed (${res.status})`);
  }

  return readSSEStream<UploadResult>(res, onProgress);
}

/* ── Query streaming ───────────────────────────────────────────── */

export function queryStream(
  body: QueryRequest,
  callbacks: QueryStreamCallbacks
): { abort: () => void } {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        callbacks.onError(data.detail ?? `Request failed (${res.status})`);
        return;
      }

      await readSSETokenStream(res, callbacks);
    } catch (e: unknown) {
      if (e instanceof Error && e.name === "AbortError") return;
      callbacks.onError(e instanceof Error ? e.message : "Network error.");
    }
  })();

  return { abort: () => controller.abort() };
}

/* ── SSE helpers ───────────────────────────────────────────────── */

/**
 * Reads an SSE stream that terminates with a final JSON payload.
 * Used by /upload — progress events are forwarded to `onProgress`, the
 * last event (no `stage`) is returned as the final result.
 */
async function readSSEStream<T>(
  res: Response,
  onProgress: (e: UploadProgressEvent) => void
): Promise<T> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let lastResult: T | null = null;

  // Prevent indefinite spinner if the stream stalls with no events.
  const readWithTimeout = async () => {
    return await Promise.race([
      reader.read(),
      new Promise<ReadableStreamReadResult<Uint8Array>>((_, reject) =>
        setTimeout(() => reject(new Error("Upload stream timed out. Please retry.")), 120000)
      ),
    ]);
  };

  while (true) {
    const { value, done } = await readWithTimeout();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (!raw) continue;

      let evt: Record<string, unknown>;
      try {
        evt = JSON.parse(raw);
      } catch {
        continue;
      }

      // A progress event has a `stage` field; the final result has `file_id`
      if (evt.stage) {
        onProgress(evt as unknown as UploadProgressEvent);
      } else if (evt.file_id) {
        lastResult = evt as unknown as T;
      } else if (evt.error) {
        onProgress({ stage: "error", message: String(evt.error) });
        throw new Error(String(evt.error));
      }
    }
  }

  if (!lastResult) throw new Error("No result returned from upload.");
  return lastResult;
}

/**
 * Reads an SSE token stream. Each event is either:
 *   { token: "..." }        — partial LLM output
 *   { sources: [...] }      — retrieval metadata
 *   { done: true }          — stream finished
 *   { error: "...", done: true } — stream error
 */
async function readSSETokenStream(
  res: Response,
  callbacks: QueryStreamCallbacks
): Promise<void> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const readWithTimeout = async () => {
    return await Promise.race([
      reader.read(),
      new Promise<ReadableStreamReadResult<Uint8Array>>((_, reject) =>
        setTimeout(() => reject(new Error("Response stream timed out. Please retry.")), 120000)
      ),
    ]);
  };

  while (true) {
    const { value, done } = await readWithTimeout();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (!raw) continue;

      let evt: Record<string, unknown>;
      try {
        evt = JSON.parse(raw);
      } catch {
        // Malformed JSON — skip silently
        continue;
      }

      if (evt.error != null) {
        callbacks.onError(String(evt.error));
        return;
      }

      if (evt.done) {
        callbacks.onDone();
        return;
      }

      if (typeof evt.token === "string" && evt.token !== "") {
        callbacks.onToken(evt.token);
      }

      if (Array.isArray(evt.sources)) {
        callbacks.onSources?.(evt.sources as Source[]);
      }
    }
  }
}
