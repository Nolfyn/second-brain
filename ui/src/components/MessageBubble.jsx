import SourceChip from "./SourceChip";

export default function MessageBubble({ role, content, sources, isStreaming }) {
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
        {/* Блок ответа */}
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

        {/* Источники */}
        {sources && sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 px-1">
            {sources.map((s, i) => (
              <SourceChip
                key={i}
                filePath={s.file_path}
                fileType={s.file_path?.split(".").pop()}
                score={s.score}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
