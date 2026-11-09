import { useEffect, useRef, useState } from "react";

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  // Автоподстройка высоты textarea под содержимое
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
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Спроси что угодно о своих заметках…"
          rows={3}
          disabled={disabled}
          className="w-full resize-none bg-transparent px-4 pt-3.5 pb-2 text-[14px] text-primary placeholder:text-muted focus:outline-none disabled:opacity-40"
        />

        <div className="flex items-center justify-between px-3 pb-3 pt-1">
          <span className="text-[11px] text-muted">Shift+Enter — новая строка · Enter — отправить</span>

          <button
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-white transition-opacity disabled:opacity-20 hover:opacity-90"
            aria-label="Отправить"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
