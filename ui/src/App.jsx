import { useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import MessageBubble from "./components/MessageBubble";
import ChatInput from "./components/ChatInput";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [recentQueries, setRecentQueries] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);
  const isSendingRef = useRef(false);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function handleSend(query) {
    if (isSendingRef.current) return;
    isSendingRef.current = true;
    setError(null);

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setRecentQueries((prev) => [query, ...prev.filter((q) => q !== query)].slice(0, 8));
    setIsStreaming(true);
    setMessages((prev) => [
      ...prev,
      { role: "ai", content: "", sources: null, isStreaming: true },
    ]);

    try {
      const response = await fetch("/api/ask", {
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
                idx === prev.length - 1
                  ? { ...m, sources: event.local }
                  : m
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
    } catch {
      setError("Can't reach the backend. Is the server running?");
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
        recentQueries={recentQueries}
        onSelectRecent={handleSend}
      />

      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-border px-6 py-3.5 flex-shrink-0">
          <span className="text-[15px] font-semibold tracking-tight text-primary">Second Brain</span>
          <span className="text-[12px] text-muted hidden sm:block">
            {isStreaming ? "Thinking…" : "Ready"}
          </span>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 ? (
            <div className="mx-auto max-w-lg pt-24 text-center">
              <p className="text-[22px] font-semibold tracking-tight text-primary mb-2">
                What do you want to know?
              </p>
              <p className="text-[14px] text-secondary leading-relaxed">
                Ask anything about your notes and documents.
                <br />
                Run <code className="rounded bg-elevated px-1.5 py-0.5 text-[12px] text-accent">brain index</code> first if you haven't already.
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
