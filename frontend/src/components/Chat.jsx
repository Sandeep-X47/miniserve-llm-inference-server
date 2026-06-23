import { useRef, useState } from "react";
import { streamChat } from "../api.js";

const TIERS = ["premium", "normal", "free"];

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [tier, setTier] = useState("normal");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef(null);

  const scrollToEnd = () => {
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  };

  const send = async () => {
    const prompt = input.trim();
    if (!prompt || busy) return;
    setInput("");
    setBusy(true);

    const userMsg = { role: "user", text: prompt };
    const botMsg = { role: "assistant", text: "", streaming: true };
    setMessages((m) => [...m, userMsg, botMsg]);
    scrollToEnd();

    await streamChat(
      { prompt, maxTokens: 64, tier },
      {
        onToken: (t) => {
          setMessages((m) => {
            const copy = [...m];
            copy[copy.length - 1] = {
              ...copy[copy.length - 1],
              text: copy[copy.length - 1].text + t,
            };
            return copy;
          });
          scrollToEnd();
        },
        onDone: () => {
          setMessages((m) => {
            const copy = [...m];
            copy[copy.length - 1] = { ...copy[copy.length - 1], streaming: false };
            return copy;
          });
          setBusy(false);
        },
        onError: (err) => {
          setMessages((m) => {
            const copy = [...m];
            copy[copy.length - 1] = {
              role: "assistant",
              text: err,
              error: true,
              streaming: false,
            };
            return copy;
          });
          setBusy(false);
        },
      }
    );
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="chat">
      <div className="chat-stream" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty">
            <p className="empty-head">Send a prompt.</p>
            <p className="empty-sub">
              It joins the request queue, gets packed into a batch by the
              scheduler, and streams back token by token. Watch it happen on the
              Console tab.
            </p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role} ${m.error ? "msg-error" : ""}`}>
            <span className="msg-role">{m.role === "user" ? "you" : "miniserve"}</span>
            <div className="msg-body">
              {m.text}
              {m.streaming && <span className="caret" />}
            </div>
          </div>
        ))}
      </div>

      <div className="composer">
        <div className="tier-row">
          <span className="tier-label">tier</span>
          {TIERS.map((t) => (
            <button
              key={t}
              className={`tier ${tier === t ? "tier-on" : ""}`}
              onClick={() => setTier(t)}
            >
              {t}
            </button>
          ))}
          <span className="tier-hint">premium jumps the queue</span>
        </div>
        <div className="input-row">
          <textarea
            className="input"
            rows={1}
            value={input}
            placeholder="Ask anything…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
          />
          <button className="send" onClick={send} disabled={busy || !input.trim()}>
            {busy ? "streaming" : "send"}
          </button>
        </div>
      </div>
    </div>
  );
}
