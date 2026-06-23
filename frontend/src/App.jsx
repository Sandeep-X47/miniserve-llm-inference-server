import { useState } from "react";
import Chat from "./components/Chat.jsx";
import Console from "./components/Console.jsx";

export default function App() {
  const [view, setView] = useState("chat");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">MiniServe</span>
          <span className="brand-sub">llm inference console</span>
        </div>
        <nav className="tabs">
          <button
            className={`tab ${view === "chat" ? "tab-on" : ""}`}
            onClick={() => setView("chat")}
          >
            Playground
          </button>
          <button
            className={`tab ${view === "console" ? "tab-on" : ""}`}
            onClick={() => setView("console")}
          >
            Console
          </button>
        </nav>
      </header>

      <main className="main">
        {view === "chat" ? <Chat /> : <Console />}
      </main>
    </div>
  );
}
