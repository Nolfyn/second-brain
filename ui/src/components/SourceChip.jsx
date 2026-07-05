const TYPE_STYLES = {
  docx: "bg-accent-glow text-accent border-accent-dim",
  doc:  "bg-accent-glow text-accent border-accent-dim",
  pdf:  "bg-elevated text-secondary border-border-bright",
  web:  "bg-elevated text-muted border-border",
};

function fileName(path) {
  if (!path) return "";
  return path.split(/[\\/]/).pop();
}

export default function SourceChip({ filePath, fileType, score, url }) {
  const style = TYPE_STYLES[fileType] || TYPE_STYLES.web;
  const display = fileType === "web" ? url : fileName(filePath);
  const label = fileType === "web" ? "Web" : fileType?.toUpperCase();

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium ${style}`}
      title={filePath || url}
    >
      <span className="opacity-60">{label}</span>
      <span className="truncate max-w-[160px] opacity-80">{display}</span>
      {typeof score === "number" && score < 1 && (
        <span className="opacity-40 tabular-nums">{score.toFixed(2)}</span>
      )}
    </span>
  );
}