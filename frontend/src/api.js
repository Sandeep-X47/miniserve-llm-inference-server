// Talks to the inference server. In dev, Vite proxies /api -> :8000.
// Override at build time with VITE_API_BASE.
const BASE = import.meta.env.VITE_API_BASE || "/api";

// POST /chat returns Server-Sent Events. EventSource only does GET, so we read
// the response body stream ourselves and parse SSE frames.
export async function streamChat({ prompt, maxTokens, tier }, { onToken, onDone, onError }) {
  try {
    const res = await fetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_tokens: maxTokens, tier }),
    });

    if (res.status === 429) {
      onError?.("Server at capacity (429). The queue is full — backpressure is working as designed. Retry shortly.");
      return;
    }
    if (!res.ok) {
      onError?.(`Request failed (${res.status}).`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.event === "token") onToken?.(ev.text);
        else if (ev.event === "done") onDone?.();
      }
    }
    onDone?.();
  } catch (e) {
    onError?.(String(e));
  }
}

export async function getStats() {
  const res = await fetch(`${BASE}/stats`);
  if (!res.ok) throw new Error(`stats ${res.status}`);
  return res.json();
}
