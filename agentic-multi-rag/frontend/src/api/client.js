import axios from "axios";

const client = axios.create({ baseURL: "/api" });

export const askQuestion = (question) =>
  client.post("/chat", { question }).then((res) => res.data);

export const ingestText = (text, source = "frontend-upload") =>
  client.post("/ingest", { text, source }).then((res) => res.data);

export const getGraph = () => client.get("/graph").then((res) => res.data);
