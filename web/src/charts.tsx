import { useMemo, useState } from "react";

const M = { l: 56, r: 16, t: 12, b: 28 };
const W = 720;
const H = 280;

function monthIdx(m: string) {
  const [y, mo] = m.split("-").map(Number);
  return y * 12 + (mo - 1);
}
const money = (v: number) => (v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M` : `$${Math.round(v / 1000)}k`);

export interface Projection {
  origin: string;
  zhvi: number;
  pred: number;
  actual: number | null;
}

/** Single-series line chart of ZHVI history with the model's projected 12-month path. */
export function HistoryChart({ history, projection }: { history: [string, number][]; projection: Projection | null }) {
  const [hover, setHover] = useState<number | null>(null);
  const pts = useMemo(() => history.map(([m, v]) => ({ i: monthIdx(m), m, v })), [history]);
  if (pts.length < 2) return <p className="muted small">No history available.</p>;
  const x0 = pts[0].i;
  const projEnd = projection ? monthIdx(projection.origin) + 12 : pts[pts.length - 1].i;
  const x1 = Math.max(pts[pts.length - 1].i, projEnd);
  const projTarget = projection ? projection.zhvi * (1 + projection.pred / 100) : null;
  const vals = pts.map((p) => p.v).concat(projTarget ? [projTarget] : []);
  const yMin = Math.min(...vals) * 0.95;
  const yMax = Math.max(...vals) * 1.05;
  const sx = (i: number) => M.l + ((i - x0) / (x1 - x0)) * (W - M.l - M.r);
  const sy = (v: number) => M.t + (1 - (v - yMin) / (yMax - yMin)) * (H - M.t - M.b);
  const path = pts.map((p, k) => `${k ? "L" : "M"}${sx(p.i).toFixed(1)},${sy(p.v).toFixed(1)}`).join("");
  const yTicks = 4;
  const ticks = Array.from({ length: yTicks + 1 }, (_, k) => yMin + ((yMax - yMin) * k) / yTicks);
  const years: number[] = [];
  for (let i = x0; i <= x1; i++) if (i % 12 === 0 && Math.floor(i / 12) % 2 === 0) years.push(i);
  const h = hover !== null ? pts[hover] : null;

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const i = x0 + ((x - M.l) / (W - M.l - M.r)) * (x1 - x0);
    let best = 0;
    let bd = Infinity;
    pts.forEach((p, k) => {
      const d = Math.abs(p.i - i);
      if (d < bd) {
        bd = d;
        best = k;
      }
    });
    setHover(best);
  }

  return (
    <div className="chart-wrap"><svg className="chart" viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label="Home value history">
      <g className="grid">
        {ticks.map((t) => (
          <line key={t} x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} />
        ))}
      </g>
      {ticks.map((t) => (
        <text key={`l${t}`} x={M.l - 8} y={sy(t) + 4} textAnchor="end">
          {money(t)}
        </text>
      ))}
      {years.map((i) => (
        <text key={i} x={sx(i)} y={H - 8} textAnchor="middle">
          {Math.floor(i / 12)}
        </text>
      ))}
      <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" />
      {projection && projTarget && (
        <>
          <line x1={sx(monthIdx(projection.origin))} x2={sx(monthIdx(projection.origin))} y1={M.t} y2={H - M.b} stroke="var(--text-3)" strokeDasharray="3 3" />
          <line
            x1={sx(monthIdx(projection.origin))} y1={sy(projection.zhvi)}
            x2={sx(projEnd)} y2={sy(projTarget)}
            stroke="var(--series-2)" strokeWidth={2.5} strokeLinecap="round"
          />
          <circle cx={sx(projEnd)} cy={sy(projTarget)} r={4} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />
        </>
      )}
      {h && (
        <g>
          <line x1={sx(h.i)} x2={sx(h.i)} y1={M.t} y2={H - M.b} stroke="var(--text-3)" strokeWidth={1} />
          <circle cx={sx(h.i)} cy={sy(h.v)} r={4.5} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
          <g transform={`translate(${Math.min(sx(h.i) + 10, W - 140)},${M.t + 4})`}>
            <rect className="tipbox" width={128} height={40} rx={6} />
            <text className="tip" x={10} y={17}>{h.m}</text>
            <text className="tip" x={10} y={33} style={{ fontWeight: 600 }}>${h.v.toLocaleString()}</text>
          </g>
        </g>
      )}
      <line className="axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
    </svg></div>
  );
}

/** Diverging horizontal bars of SHAP contributions (blue = pushes prediction up, red = down). */
export function ContributionBars({ items, format }: { items: { label: string; value: string; shap: number }[]; format?: (n: number) => string }) {
  const max = Math.max(0.5, ...items.map((c) => Math.abs(c.shap)));
  const f = format ?? ((n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(2)}`);
  return (
    <div className="bars" role="table" aria-label="Feature contributions">
      {items.map((c) => {
        const w = (Math.abs(c.shap) / max) * 42;
        const pos = c.shap >= 0;
        return (
          <div key={c.label} style={{ display: "contents" }}>
            <div className="f">{c.label}</div>
            <div className="v">{c.value}</div>
            <div className="track" title={`${c.label}: ${f(c.shap)} pp`}>
              <div className={`bar ${pos ? "pos" : "neg"}`} style={pos ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
              <span className="n" style={pos ? { left: `calc(50% + ${w}% + 6px)` } : { left: "calc(50% + 6px)" }}>{f(c.shap)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Two-series line chart: mean prediction vs mean actual by origin. */
export function OriginChart({ rows }: { rows: { origin: string; pred_mean: number; actual_mean: number | null }[] }) {
  const w = 720;
  const h = 200;
  const m = { l: 40, r: 36, t: 12, b: 28 };
  const vals = rows.flatMap((r) => [r.pred_mean, r.actual_mean ?? r.pred_mean]);
  const yMin = Math.min(0, ...vals) - 1;
  const yMax = Math.max(...vals) + 1;
  const sx = (k: number) => m.l + (k / Math.max(1, rows.length - 1)) * (w - m.l - m.r);
  const sy = (v: number) => m.t + (1 - (v - yMin) / (yMax - yMin)) * (h - m.t - m.b);
  const line = (sel: (r: (typeof rows)[number]) => number | null) =>
    rows
      .map((r, k) => {
        const v = sel(r);
        return v === null ? null : `${k && sel(rows[k - 1]) !== null ? "L" : "M"}${sx(k).toFixed(1)},${sy(v).toFixed(1)}`;
      })
      .filter(Boolean)
      .join("");
  const ticks = [yMin, (yMin + yMax) / 2, yMax].map((t) => Math.round(t));
  return (
    <div className="chart-wrap"><svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Mean prediction vs actual by origin">
      <g className="grid">{ticks.map((t) => <line key={t} x1={m.l} x2={w - m.r} y1={sy(t)} y2={sy(t)} />)}</g>
      {ticks.map((t) => <text key={`t${t}`} x={m.l - 8} y={sy(t) + 4} textAnchor="end">{t}%</text>)}
      <line x1={m.l} x2={w - m.r} y1={sy(0)} y2={sy(0)} stroke="var(--text-3)" strokeWidth={1} />
      <path d={line((r) => r.pred_mean)} fill="none" stroke="var(--series-2)" strokeWidth={2} />
      <path d={line((r) => r.actual_mean)} fill="none" stroke="var(--series-1)" strokeWidth={2} />
      {rows.map((r, k) => (
        <g key={r.origin}>
          <circle cx={sx(k)} cy={sy(r.pred_mean)} r={4} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2}><title>{r.origin}: predicted {r.pred_mean.toFixed(1)}%</title></circle>
          {r.actual_mean !== null && <circle cx={sx(k)} cy={sy(r.actual_mean)} r={4} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2}><title>{r.origin}: actual {r.actual_mean.toFixed(1)}%</title></circle>}
          <text x={sx(k)} y={h - 8} textAnchor="middle">{r.origin}</text>
        </g>
      ))}
    </svg></div>
  );
}
