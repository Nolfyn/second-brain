import { useEffect, useState } from "react";

const FILTERS = [
  { key: "all",  label: "All sources" },
  { key: "pdf",  label: "PDFs" },
  { key: "docx", label: "Word docs" },
  { key: "web",  label: "Web" },
];

export default function Sidebar({ activeFilter, onFilterChange, recentQueries, onSelectRecent }) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  return (
    <aside className="flex h-full w-[260px] flex-shrink-0 flex-col border-r border-border bg-surface">
      <div className="px-4 pt-5 pb-4 border-b border-border">
        <p className="text-[10px] font-semibold tracking-widest uppercase text-muted mb-3">Sources</p>
        {FILTERS.map((f) => (
          <button
            key={f.key}
            onClick={() => onFilterChange(f.key)}
            className={`w-full text-left px-3 py-2 rounded-lg text-[13px] mb-0.5 transition-colors ${
              activeFilter === f.key
                ? "bg-accent-glow text-accent font-medium"
                : "text-secondary hover:text-primary hover:bg-elevated"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        <p className="text-[10px] font-semibold tracking-widest uppercase text-muted mb-3">Recent</p>
        {recentQueries.length === 0 ? (
          <p className="text-[12px] text-muted px-1">Questions appear here</p>
        ) : (
          recentQueries.map((q, i) => (
            <button
              key={i}
              onClick={() => onSelectRecent(q)}
              className="block w-full text-left px-3 py-1.5 rounded-lg text-[12.5px] text-secondary hover:text-primary hover:bg-elevated truncate transition-colors"
              title={q}
            >
              {q}
            </button>
          ))
        )}
      </div>

      {stats && (
        <div className="border-t border-border px-4 py-4">
          <p className="text-[10px] font-semibold tracking-widest uppercase text-muted mb-2">Index</p>
          <div className="space-y-1">
            <div className="flex justify-between text-[12px]">
              <span className="text-muted">Files</span>
              <span className="text-secondary tabular-nums">{stats.total_files}</span>
            </div>
            <div className="flex justify-between text-[12px]">
              <span className="text-muted">Chunks</span>
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