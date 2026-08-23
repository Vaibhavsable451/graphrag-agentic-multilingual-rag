export default function MessageBubble({ role, content, meta }) {
  const isUser = role === "user";
  return (
    <div className={`bubble-row ${isUser ? "bubble-row--user" : ""}`}>
      <div className={`bubble ${isUser ? "bubble--user" : "bubble--bot"}`}>
        <p>{content}</p>
        {meta && !isUser && (
          <div className="bubble-meta">
            <span className="tag">{meta.language}</span>
            <span className="tag">route: {meta.route}</span>
            {meta.sources?.length > 0 && <span className="tag">{meta.sources.length} sources</span>}
            {meta.graph_facts?.length > 0 && <span className="tag">{meta.graph_facts.length} graph facts</span>}
          </div>
        )}
      </div>
    </div>
  );
}
