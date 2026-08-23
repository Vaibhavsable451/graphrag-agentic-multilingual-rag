import { useState } from "react";
import { ingestText } from "../api/client";

export default function Sidebar({ onIngested }) {
  const [text, setText] = useState("");
  const [status, setStatus] = useState("");

  async function handleIngest() {
    if (!text.trim()) return;
    setStatus("Ingesting…");
    try {
      const res = await ingestText(text);
      setStatus(`Added ${res.entities_found} entities to the graph.`);
      setText("");
      onIngested?.();
    } catch {
      setStatus("Ingestion failed.");
    }
  }

  return (
    <aside className="sidebar">
      <h2>Agentic Multi-RAG</h2>
      <p className="sidebar-sub">GraphRAG · Agentic RAG · Multilingual RAG</p>

      <div className="ingest-box">
        <label>Add knowledge</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste text to ingest into the vector store and knowledge graph…"
          rows={6}
        />
        <button onClick={handleIngest}>Ingest</button>
        {status && <p className="status">{status}</p>}
      </div>

      <div className="stack-list">
        <p className="stack-title">Stack</p>
        <ul>
          <li>LangChain + LangGraph</li>
          <li>Groq · OpenRouter · Gemini</li>
          <li>Pinecone vector store</li>
          <li>MCP tool server</li>
          <li>FastAPI backend</li>
        </ul>
      </div>
    </aside>
  );
}
