import { useEffect, useState } from "react";

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
}) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  return (
    <aside className="flex h-full w-[260px] flex-shrink-0 flex-col border-r border-border bg-surface">
      <div className="px-4 pt-4 pb-2">
        <button
          onClick={onNewChat}
          className="w-full rounded-lg bg-accent px-3 py-2 text-[13px] font-medium text-white transition-opacity hover:opacity-90"
        >
          + Новый чат
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-2">
        <p className="text-[10px] font-semibold tracking-widest uppercase text-muted mb-2 px-1">Чаты</p>
        {sessions.length === 0 ? (
          <p className="text-[12px] text-muted px-1">Чаты появятся после первого вопроса</p>
        ) : (
          sessions.map((s) => (
            <div
              key={s.id}
              className={`group flex items-center rounded-lg mb-0.5 transition-colors ${
                activeSessionId === s.id ? "bg-accent-glow" : "hover:bg-elevated"
              }`}
            >
              <button
                onClick={() => onSelectSession(s.id)}
                className={`min-w-0 flex-1 text-left px-3 py-1.5 text-[12.5px] truncate ${
                  activeSessionId === s.id ? "text-accent font-medium" : "text-secondary hover:text-primary"
                }`}
                title={s.title}
              >
                {s.title}
              </button>
              <button
                onClick={() => onDeleteSession(s.id)}
                className="hidden group-hover:block pr-2 text-muted hover:text-warn text-[14px] leading-none"
                title="Удалить чат"
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>

      {stats && (
        <div className="border-t border-border px-4 py-4">
          <p className="text-[10px] font-semibold tracking-widest uppercase text-muted mb-2">Индекс</p>
          <div className="space-y-1">
            <div className="flex justify-between text-[12px]">
              <span className="text-muted">Файлы</span>
              <span className="text-secondary tabular-nums">{stats.total_files}</span>
            </div>
            <div className="flex justify-between text-[12px]">
              <span className="text-muted">Чанки</span>
              <span className="text-secondary tabular-nums">{stats.total_chunks?.toLocaleString()}</span>
            </div>
            {stats.by_type && Object.entries(stats.by_type).map(([type, count]) => (
              <div key={type} className="flex justify-between text-[11px] pl-2">
                <span className="text-muted">{type}</span>
                <span className="text-muted tabular-nums">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}
