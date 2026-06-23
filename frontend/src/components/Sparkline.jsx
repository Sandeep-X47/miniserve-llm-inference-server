// Dependency-free sparkline. Keeps the bundle tiny — no charting library for
// what is fundamentally a polyline.
export default function Sparkline({ data, width = 260, height = 56, color = "var(--accent)" }) {
  if (!data || data.length < 2) {
    return <svg width={width} height={height} className="spark" />;
  }
  const max = Math.max(...data, 1);
  const stepX = width / (data.length - 1);
  const points = data
    .map((v, i) => `${(i * stepX).toFixed(1)},${(height - (v / max) * (height - 6) - 3).toFixed(1)}`)
    .join(" ");
  const area = `0,${height} ${points} ${width},${height}`;

  return (
    <svg width={width} height={height} className="spark" preserveAspectRatio="none" viewBox={`0 0 ${width} ${height}`}>
      <polygon points={area} fill={color} opacity="0.10" />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}
