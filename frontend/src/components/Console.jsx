import { useEffect, useRef, useState } from "react";
import { getStats } from "../api.js";
import Sparkline from "./Sparkline.jsx";

const POLL_MS = 1000;
const HISTORY = 40;

export default function Console() {
  const [stats, setStats] = useState(null);
  const [tpsHistory, setTpsHistory] = useState([]);
  const [connected, setConnected] = useState(true);
  const timer = useRef(null);

  useEffect(() => {
    const tick = async () => {
      try {
        const s = await getStats();
        setStats(s);
        setConnected(true);
        setTpsHistory((h) => [...h, s.tokens_per_sec].slice(-HISTORY));
      } catch {
        setConnected(false);
      }
    };
    tick();
    timer.current = setInterval(tick, POLL_MS);
    return () => clearInterval(timer.current);
  }, []);

  if (!stats) {
    return (
      <div className="console">
        <div className="console-empty">
          {connected ? "Connecting to server…" : "No server on /api. Start the backend (uvicorn app.main:app)."}
        </div>
      </div>
    );
  }

  const cap = stats.queue_capacity || 1;
  const queuePct = Math.min(100, (stats.queue_depth / cap) * 100);
  const maxBatch = stats.max_batch_size || 16;
  const slots = Array.from({ length: maxBatch }, (_, i) => i < stats.running_sequences);

  return (
    <div className="console">
      <div className="console-head">
        <div className="console-title">
          telemetry
          <span className={`dot ${connected ? "dot-ok" : "dot-bad"}`} />
        </div>
        <div className="console-meta">
          engine <b>{stats.engine}</b> · batching <b>{stats.batching}</b> · max batch <b>{maxBatch}</b>
        </div>
      </div>

      {/* Signature element: watch the scheduler pack the running batch live. */}
      <section className="panel batch-panel">
        <div className="panel-label">running batch · {stats.running_sequences}/{maxBatch} slots</div>
        <div className="slots">
          {slots.map((on, i) => (
            <span key={i} className={`slot ${on ? "slot-on" : ""}`} />
          ))}
        </div>
        <div className="panel-foot">
          Each slot is a sequence the GPU is decoding this step. In continuous
          batching, finished slots are refilled from the queue immediately.
        </div>
      </section>

      <div className="grid">
        <Metric label="tokens / sec" value={stats.tokens_per_sec} accent />
        <Metric label="requests / sec" value={stats.requests_per_sec} />
        <Metric label="batch size" value={stats.last_batch_size} />
        <Metric label="latency p50" value={`${stats.latency_p50_ms} ms`} />
        <Metric label="latency p95" value={`${stats.latency_p95_ms} ms`} />
        <Metric label="completed" value={stats.completed} />
      </div>

      <section className="panel">
        <div className="panel-label">throughput · last {HISTORY}s</div>
        <Sparkline data={tpsHistory} width={520} height={70} />
      </section>

      <section className="panel">
        <div className="panel-label">
          queue depth · {stats.queue_depth}/{cap}
          {stats.rejected > 0 && <span className="rejected"> · {stats.rejected} rejected (429)</span>}
        </div>
        <div className="bar">
          <div className={`bar-fill ${queuePct > 80 ? "bar-hot" : ""}`} style={{ width: `${queuePct}%` }} />
        </div>
        <div className="panel-foot">
          When this hits capacity the server returns 429 instead of buffering
          forever — that's backpressure.
        </div>
      </section>
    </div>
  );
}

function Metric({ label, value, accent }) {
  return (
    <div className={`metric ${accent ? "metric-accent" : ""}`}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
