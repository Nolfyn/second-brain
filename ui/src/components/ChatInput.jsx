import { useEffect, useRef, useState } from "react";

export default function ChatInput({ onSend, webSearch, onToggleWebSearch, agentMode, onToggleAgent, disabled }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  // Auto-expand textarea as content grows
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [value]);

  function handleSubmit(e) {
    e.preventDefault();
    if (disabled) return;
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="rounded-2xl border border-border bg-surface transition-colors focus-within:border-border-bright">
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about your notes…"
          rows={3}
          disabled={disabled}
          className="w-full resize-none bg-transparent px-4 pt-3.5 pb-2 text-[14px] text-primary placeholder:text-muted focus:outline-none disabled:opacity-40"
        />

        {/* Bottom toolbar */}
        <div className="flex items-center justify-between px-3 pb-3 pt-1">
          <div className="flex items-center gap-2">
            {/* Agent toggle */}
            <button
              onClick={onToggleAgent}
              className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors ${
                agentMode
                  ? "bg-accent text-white"
                  : "bg-elevated text-secondary hover:text-primary border border-border"
              }`}
              title="Let the model choose which search tools to use"
            >
              <span className={`h-1.5 w-1.5 rounded-full ${agentMode ? "bg-white" : "bg-muted"}`} />
              Agent
            </button>

            {/* Web search toggle */}
            <button
              onClick={onToggleWebSearch}
              disabled={agentMode}
              className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors ${
                webSearch && !agentMode
                  ? "bg-elevated text-success border border-success/30"
                  : "bg-elevated text-secondary hover:text-primary border border-border"
              } disabled:opacity-30 disabled:cursor-not-allowed`}
              title={agentMode ? "Web search is controlled by the agent" : "Include live web results"}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${webSearch && !agentMode ? "bg-success" : "bg-muted"}`} />
              Web
            </button>
          </div>

          {/* Send button */}
          <button
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-white transition-opacity disabled:opacity-20 hover:opacity-90"
            aria-label="Send"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      </div>
      <p className="mt-1.5 text-center text-[11px] text-muted">
        Shift+Enter for new line · Enter to send
      </p>
    </div>
  );
}