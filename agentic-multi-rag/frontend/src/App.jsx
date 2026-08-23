import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <ChatWindow />
    </div>
  );
}
