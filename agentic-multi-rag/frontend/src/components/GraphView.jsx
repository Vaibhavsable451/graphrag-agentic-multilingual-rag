import { useEffect, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { getGraph } from "../api/client";

export default function GraphView() {
  const [data, setData] = useState({ nodes: [], links: [] });

  async function refresh() {
    const g = await getGraph();
    setData({
      nodes: g.nodes.map((n) => ({ id: n.id, type: n.type })),
      links: g.edges.map((e) => ({ source: e.source, target: e.target, relation: e.relation })),
    });
  }

  useEffect(() => {
    refresh();
  }, []);

  if (data.nodes.length === 0) {
    return <div className="graph-empty">Ingest some text to see the knowledge graph here.</div>;
  }

  return (
    <div className="graph-view">
      <ForceGraph2D
        graphData={data}
        nodeLabel={(n) => `${n.id} (${n.type})`}
        linkLabel={(l) => l.relation}
        nodeAutoColorBy="type"
        linkDirectionalArrowLength={4}
        height={320}
      />
    </div>
  );
}
