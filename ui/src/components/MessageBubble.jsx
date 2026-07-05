import SourceChip from "./SourceChip";

// Human-readable tool names for display
const TOOL_LABELS = {
  keyword_search:  "Searching index",
  semantic_search: "Semantic search",
  web_search:      "Web search",
  get_full_file:   "Reading file",
  list_files:      "Listing files",
};

export default function MessageBubble({ role, content, sources, toolActivity, isStreaming }) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[72%] rounded-2xl rounded-br-sm bg-elevated border border-border px-4 py-3 text-[14px] text-primary leading-relaxed">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] space-y-2">

        {/* Tool activity -- signature element: translucent pills showing agent thinking */}
        {toolActivity && toolActivity.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-1">
            {toolActivity.map((event, i) => {
              const isLast = i === toolActivity.length - 1 && isStreaming;
              const label = TOOL_LABELS[event.tool] || event.tool;
              const arg = Object.values(event.arguments || {})[0] || "";
              return (
                <span
                  key={i}
                  className={`inline-flex items-center gap-1.5 rounded-full border border-border-bright bg-elevated px-3 py-1 text-[11.5px] text-secondary ${isLast ? "thinking-dot" : ""}`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${isLast ? "bg-accent thinking-dot" : "bg-success"}`} />
                  <span className="font-medium">{label}</span>
                  {arg && <span className="text-muted truncate max-w-[180px]">{arg}</span>}
                </span>
              );
            })}
          </div>
        )}

        {/* Answer bubble */}
        {(content || isStreaming) && (
          <div className="rounded-2xl rounded-bl-sm bg-elevated border border-border px-5 py-4">
            <p className="text-[14.5px] text-primary leading-[1.8] whitespace-pre-wrap">
              {content}
              {isStreaming && !content && (
                <span className="inline-flex gap-1 ml-1">
                  <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-muted inline-block" />
                  <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-muted inline-block" />
                  <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-muted inline-block" />
                </span>
              )}
              {isStreaming && content && (
                <span className="ml-0.5 inline-block h-[15px] w-[2px] translate-y-[2px] animate-pulse bg-accent" />
              )}
            </p>
          </div>
        )}

        {/* Source chips */}
        {sources && (sources.local?.length > 0 || sources.web?.length > 0) && (
          <div className="flex flex-wrap gap-1.5 px-1">
            {sources.local?.map((s, i) => (
              <SourceChip
                key={`l-${i}`}
                filePath={s.file_path}
                fileType={s.file_path?.split(".").pop()}
                score={s.score}
              />
            ))}
            {sources.web?.map((s, i) => (
              <SourceChip key={`w-${i}`} fileType="web" url={s.url} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}