import { useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import MessageBubble from "./components/MessageBubble";
import ChatInput from "./components/ChatInput";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);
  const isSendingRef = useRef(false);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function refreshSessions() {
    try {
      const r = await fetch("/api/sessions");
      if (r.ok) setSessions(await r.json());
    } catch {
      /* backend недоступен - список не критичен */
    }
  }

  useEffect(() => {
    refreshSessions();
  }, []);

  async function ensureSession() {
    if (sessionId) return sessionId;
    const r = await fetch("/api/sessions", { method: "POST" });
    const { id } = await r.json();
    setSessionId(id);
    refreshSessions();
    return id;
  }

  async function loadSession(id) {
    if (isSendingRef.current) return;
    const r = await fetch(`/api/sessions/${id}`);
    if (!r.ok) return;
    const session = await r.json();
    setSessionId(id);
    setMessages(
      session.messages.map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources || null,
        isStreaming: false,
      }))
    );
    setError(null);
  }

  function newChat() {
    if (isSendingRef.current) return;
    setSessionId(null);
    setMessages([]);
    setError(null);
  }

  async function deleteSession(id) {
    await fetch(`/api/sessions/${id}`, { method: "DELETE" });
    if (id === sessionId) newChat();
    refreshSessions();
  }

  async function handleSend(query) {
    if (isSendingRef.current) return;
    isSendingRef.current = true;
    setError(null);

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setIsStreaming(true);
    setMessages((prev) => [
      ...prev,
      { role: "ai", content: "", sources: null, isStreaming: true },
    ]);

    try {
      const sid = await ensureSession();
      const response = await fetch(`/api/sessions/${sid}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!response.ok || !response.body) throw new Error(`${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);

          if (event.type === "sources") {
            setMessages((prev) =>
              prev.map((m, idx) =>
                idx === prev.length - 1 ? { ...m, sources: event.local } : m
              )
            );
          } else if (event.type === "token") {
            setMessages((prev) =>
              prev.map((m, idx) =>
                idx === prev.length - 1
                  ? { ...m, content: m.content + event.content }
                  : m
              )
            );
          }
        }
      }
      refreshSessions();
    } catch {
      setError("Не удаётся связаться с бэкендом. Сервер запущен?");
    } finally {
      isSendingRef.current = false;
      setIsStreaming(false);
      setMessages((prev) =>
        prev.map((m, idx) => (idx === prev.length - 1 ? { ...m, isStreaming: false } : m))
      );
    }
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-base text-primary">
      <Sidebar
        sessions={sessions}
        activeSessionId={sessionId}
        onSelectSession={loadSession}
        onNewChat={newChat}
        onDeleteSession={deleteSession}
      />

      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-border px-6 py-3.5 flex-shrink-0">
          <span className="text-[15px] font-semibold tracking-tight text-primary">Second Brain</span>
          <span className="text-[12px] text-muted hidden sm:block">
            {isStreaming ? "Думает…" : "Готово"}
          </span>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 ? (
            <div className="mx-auto max-w-lg pt-24 text-center">
              <p className="text-[22px] font-semibold tracking-tight text-primary mb-2">
                О чём спросить?
              </p>
              <p className="text-[14px] text-secondary leading-relaxed">
                Спроси что угодно о своих заметках и документах.
                <br />
                Сначала запусти <code className="rounded bg-elevated px-1.5 py-0.5 text-[12px] text-accent">brain index</code>, если ещё этого не делал.
              </p>
            </div>
          ) : (
            <div className="mx-auto flex max-w-chat flex-col gap-6">
              {messages.map((m, i) => (
                <MessageBubble
                  key={i}
                  role={m.role}
                  content={m.content}
                  sources={m.sources}
                  isStreaming={m.isStreaming}
                />
              ))}
            </div>
          )}

          {error && (
            <div className="mx-auto mt-4 max-w-chat rounded-xl border border-warn/20 bg-warn/5 px-4 py-3 text-[13px] text-warn">
              {error}
            </div>
          )}
        </div>

        {/* Input */}
        <div className="mx-auto w-full max-w-chat flex-shrink-0">
          <ChatInput onSend={handleSend} disabled={isStreaming} />
        </div>
      </div>
    </div>
  );
}
