export default function Pill({ kind, children }) {
  return <span className={`pill ${kind}`}>{children}</span>;
}
