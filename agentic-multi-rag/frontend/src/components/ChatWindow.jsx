import { useState, useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import GraphView from "./GraphView";
import { askQuestion } from "../api/client";

export default function ChatWindow() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Ask me anything — I'll route your question through vector search, graph traversal, or both, in whatever language you ask.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showGraph, setShowGraph] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await askQuestion(question);
      setMessages((m) => [...m, { role: "assistant", content: res.answer, meta: res }]);
    } catch {
      setMessages((m) => [...m, { role: "assistant", content: "Something went wrong reaching the backend." }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-window">
      <div className="chat-header">
        <h3>Agent Console</h3>
        <button className="ghost-btn" onClick={() => setShowGraph((s) => !s)}>
          {showGraph ? "Hide graph" : "Show knowledge graph"}
        </button>
      </div>

      {showGraph && <GraphView />}

      <div className="chat-log">
        {messages.map((m, i) => (
          <MessageBubble key={i} role={m.role} content={m.content} meta={m.meta} />
        ))}
        {loading && <div className="bubble bubble--bot bubble--typing">Thinking…</div>}
        <div ref={endRef} />
      </div>

      <div className="chat-input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask a question, in any language…"
        />
        <button onClick={handleSend} disabled={loading}>Send</button>
      </div>
    </div>
  );
}
