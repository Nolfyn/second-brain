const TYPE_STYLES = {
  docx: "bg-accent-glow text-accent border-accent-dim",
  doc:  "bg-accent-glow text-accent border-accent-dim",
  pdf:  "bg-elevated text-secondary border-border-bright",
};

function fileName(path) {
  if (!path) return "";
  return path.split(/[\\/]/).pop();
}

export default function SourceChip({ filePath, fileType, score }) {
  const style = TYPE_STYLES[fileType] || "bg-elevated text-muted border-border";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium ${style}`}
      title={filePath}
    >
      <span className="opacity-60">{fileType?.toUpperCase()}</span>
      <span className="truncate max-w-[160px] opacity-80">{fileName(filePath)}</span>
      {typeof score === "number" && score < 1 && (
        <span className="opacity-40 tabular-nums">{score.toFixed(2)}</span>
      )}
    </span>
  );
}
