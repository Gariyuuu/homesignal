import { useEffect, useMemo, useState } from "react";
import { ContributionBars, HistoryChart, OriginChart } from "./charts";
import { fmtValue, friendly, loadIndex, loadMeta, loadZip, pct, type IndexRow, type Meta, type ZipRecord } from "./data";

const REPO = "https://github.com/Gariyuuu/homesignal";

function useRoute() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const go = (to: string) => {
    window.history.pushState({}, "", to);
    setPath(to);
    window.scrollTo({ top: 0 });
  };
  return { path, go };
}

function Disclaimer() {
  return (
    <div className="disclaimer">
      <b>Educational project, not advice.</b> This is a statistical model with large errors (holdout MAE ≈ 5.8 pp; it did
      not beat a zero-growth baseline on 2024–25 origins). Not financial, investment, lending or appraisal advice.
    </div>
  );
}

function Search({ index, go, autoFocus }: { index: IndexRow[]; go: (to: string) => void; autoFocus?: boolean }) {
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const results = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return [];
    const isZip = /^\d+$/.test(s);
    const out: IndexRow[] = [];
    for (const r of index) {
      if (isZip ? r[0].startsWith(s) : `${r[1]}, ${r[2]} ${r[3]}`.toLowerCase().includes(s)) {
        out.push(r);
        if (out.length >= 12) break;
      }
    }
    return out;
  }, [q, index]);
  return (
    <div className="search">
      <input
        autoFocus={autoFocus}
        placeholder="Search a ZIP code or city — e.g. 78702 or Austin"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setActive(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") setActive((a) => Math.min(a + 1, results.length - 1));
          if (e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
          if (e.key === "Enter" && results[active]) go(`/zip/${results[active][0]}`);
        }}
        aria-label="Search ZIP code or city"
      />
      {results.length > 0 && (
        <ul>
          {results.map((r, i) => (
            <li key={r[0]} className={i === active ? "active" : ""} onMouseDown={() => go(`/zip/${r[0]}`)}>
              <span>
                <span className="z">{r[0]}</span> &nbsp;{r[1]}, {r[2]} <span className="muted small">{r[3]}</span>
              </span>
              <span className="p">{pct(r[4])}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Home({ meta, index, go }: { meta: Meta; index: IndexRow[]; go: (to: string) => void }) {
  const hm = meta.holdout;
  const fm = hm.final_model;
  const list = (rows: Meta["top"]) => (
    <table>
      <thead>
        <tr>
          <th>ZIP</th>
          <th>Place</th>
          <th className="num">ZHVI</th>
          <th className="num">Predicted 12m</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.zip} className="link" onClick={() => go(`/zip/${r.zip}`)}>
            <td style={{ fontFamily: "var(--mono)" }}>{r.zip}</td>
            <td>
              {r.city}, {r.state}
            </td>
            <td className="num">${r.zhvi.toLocaleString()}</td>
            <td className="num">{pct(r.pred)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
  return (
    <>
      <h1>12-month home value growth, by ZIP code</h1>
      <p className="lead">
        A LightGBM model trained on public data (Zillow, Census ACS, Freddie Mac, BLS) predicts how much the Zillow Home
        Value Index of each of {meta.n_zips.toLocaleString()} ZIP codes will change over the next 12 months, and explains
        each prediction with SHAP. Latest prediction origin: <b>{meta.latest_origin}</b>.
      </p>
      <Search index={index} go={go} autoFocus />

      <h2>How good is it? Honestly: not very</h2>
      <div className="grid cols-3">
        <div className="card stat">
          <span className="label">Holdout MAE, final model</span>
          <span className="value">{fm.mae?.toFixed(2)} pp</span>
          <span className="sub">
            origins {meta.holdout_origins[0].slice(0, 7)} → {meta.holdout_origins[1].slice(0, 7)}, {meta.holdout_rows.toLocaleString()} ZIP-quarters
          </span>
        </div>
        <div className="card stat">
          <span className="label">Holdout MAE, "predict zero"</span>
          <span className="value">{hm.baseline_zero.mae?.toFixed(2)} pp</span>
          <span className="sub">a naive baseline the model failed to beat</span>
        </div>
        <div className="card stat">
          <span className="label">Ranking skill (Spearman within origin)</span>
          <span className="value">{fm.cs_spearman?.toFixed(2)}</span>
          <span className="sub">≈0.31 on the 2016–23 backtest, ≈0.07 since 2022</span>
        </div>
      </div>
      <p className="note" style={{ marginTop: 12 }}>
        The model learned the 2020–21 low-rate boom and extrapolated it: on the untouched holdout it predicted 6–9 % growth
        in a period that delivered 1–3 %. The headline number was not tuned away — it is reported as-is. Full per-fold
        tables, ablations and the discussion are in the{" "}
        <a href={`${REPO}/blob/main/reports/evaluation.md`} target="_blank" rel="noreferrer">
          evaluation report
        </a>{" "}
        and{" "}
        <a href={`${REPO}/blob/main/reports/model_card.md`} target="_blank" rel="noreferrer">
          model card
        </a>
        .
      </p>

      <h2>Mean prediction vs. realised growth, by origin</h2>
      <div className="card">
        <div className="legend">
          <span>
            <i style={{ background: "var(--series-2)" }} />
            mean prediction
          </span>
          <span>
            <i style={{ background: "var(--series-1)" }} />
            mean actual (where the 12 months have elapsed)
          </span>
        </div>
        <OriginChart rows={meta.by_origin} />
        <p className="note">Averages across all {meta.n_zips.toLocaleString()} ZIPs at each quarterly origin. The gap is the model's level bias.</p>
      </div>

      <h2>What drives predictions (global SHAP importance)</h2>
      <div className="card">
        <ContributionBars items={meta.importance.map(([f, v]) => ({ label: friendly(f), value: "", shap: v }))} format={(n) => n.toFixed(2)} />
        <p className="note">Mean |SHAP| in percentage points over a 20,000-row training sample. The mortgage rate dominates — the root of the regime-extrapolation problem.</p>
      </div>

      <div className="grid cols-2" style={{ marginTop: 32 }}>
        <div>
          <h2 style={{ marginTop: 0 }}>Highest predicted growth ({meta.latest_origin})</h2>
          <div className="card pad-0">{list(meta.top)}</div>
        </div>
        <div>
          <h2 style={{ marginTop: 0 }}>Lowest predicted growth ({meta.latest_origin})</h2>
          <div className="card pad-0">{list(meta.bottom)}</div>
        </div>
      </div>
      <p className="note">Extreme predictions are usually small ZIPs with noisy indices. Treat these as rankings, not magnitudes.</p>

      <h2>Method in one paragraph</h2>
      <p className="note" style={{ maxWidth: "80ch" }}>
        Target: percent change in the raw (not seasonally adjusted) mid-tier ZHVI from month <i>t</i> to <i>t</i>+12. Every
        feature uses only data public at <i>t</i>: momentum and valuation from the index, ACS demographics assigned to the
        months each vintage was actually released, macro series with publication lags, and coarse geography (no raw
        identifiers). Expanding-window backtest 2016–2023 with a 12-month gap, then one evaluation on a holdout of
        2024–25 origins. Leakage is tested by scrambling all inputs after a cutoff and asserting nothing before it changes.{" "}
        <a href={`${REPO}/blob/main/docs/methodology.md`} target="_blank" rel="noreferrer">
          Methodology
        </a>
        .
      </p>
    </>
  );
}

function ZipPage({ zip, meta, go }: { zip: string; meta: Meta; go: (to: string) => void }) {
  const [rec, setRec] = useState<ZipRecord | null | undefined>(undefined);
  const [origin, setOrigin] = useState(meta.latest_origin);
  useEffect(() => {
    setRec(undefined);
    loadZip(zip).then(setRec);
  }, [zip]);
  if (rec === undefined) return <p className="muted">Loading {zip}…</p>;
  if (rec === null)
    return (
      <>
        <h1>ZIP {zip}</h1>
        <p className="note">
          No model data for this ZIP (Zillow does not publish an index for it, or it has too little history).{" "}
          <button className="link" onClick={() => go("/")}>
            Back to search
          </button>
        </p>
      </>
    );
  const p = rec.preds.find((x) => x.origin === origin) ?? rec.preds[rec.preds.length - 1];
  const err = meta.holdout.final_model.mae ?? 5.8;
  return (
    <>
      <p className="small">
        <button className="link" onClick={() => go("/")}>
          ← All ZIPs
        </button>
      </p>
      <h1>
        {rec.zip} · {rec.city}, {rec.state}
      </h1>
      <p className="lead">
        {rec.metro || "Non-metro"} {rec.county && `· ${rec.county}`}
      </p>
      <div className="row" style={{ marginBottom: 16 }}>
        <label className="small muted" htmlFor="origin">
          Prediction origin
        </label>
        <select id="origin" value={p.origin} onChange={(e) => setOrigin(e.target.value)}>
          {rec.preds.map((x) => (
            <option key={x.origin} value={x.origin}>
              {x.origin}
              {x.actual !== null ? " (outcome known)" : ""}
            </option>
          ))}
        </select>
      </div>
      <div className="grid cols-3">
        <div className="card stat">
          <span className="label">ZHVI at origin ({p.origin})</span>
          <span className="value">${p.zhvi.toLocaleString()}</span>
          <span className="sub">trailing 12-month growth {pct(p.growth_12m)}</span>
        </div>
        <div className="card stat">
          <span className="label">Predicted next 12 months</span>
          <span className="value" style={{ color: "var(--series-2)" }}>
            {pct(p.pred)}
          </span>
          <span className="sub">typical error ≈ ±{err.toFixed(1)} pp — read as a ranking, not a forecast</span>
        </div>
        <div className="card stat">
          <span className="label">Actual 12-month growth</span>
          <span className="value">{p.actual === null ? "—" : pct(p.actual)}</span>
          <span className="sub">{p.actual === null ? "not yet observed" : `error ${pct(p.pred - p.actual)}`}</span>
        </div>
      </div>

      <h2>Home value history</h2>
      <div className="card">
        <div className="legend">
          <span>
            <i style={{ background: "var(--series-1)" }} />
            ZHVI (mid-tier, raw)
          </span>
          <span>
            <i style={{ background: "var(--series-2)" }} />
            model's projected 12-month path from {p.origin}
          </span>
        </div>
        <HistoryChart history={rec.history} projection={{ origin: p.origin, zhvi: p.zhvi, pred: p.pred, actual: p.actual }} />
      </div>

      <h2>Why this prediction?</h2>
      <div className="card">
        <ContributionBars items={p.shap.map(([f, v, s]) => ({ label: friendly(f), value: fmtValue(f, v), shap: s }))} />
        <p className="note">
          Top SHAP contributions in percentage points. The model's average prediction is the starting point; blue bars push
          this ZIP's prediction up, red bars push it down.
        </p>
      </div>

      <h2>All bundled origins</h2>
      <div className="card pad-0">
        <table>
          <thead>
            <tr>
              <th>Origin</th>
              <th className="num">ZHVI</th>
              <th className="num">Predicted</th>
              <th className="num">Actual</th>
              <th className="num">Error</th>
            </tr>
          </thead>
          <tbody>
            {rec.preds.map((x) => (
              <tr key={x.origin} className="link" onClick={() => setOrigin(x.origin)}>
                <td style={{ fontFamily: "var(--mono)" }}>{x.origin}</td>
                <td className="num">${x.zhvi.toLocaleString()}</td>
                <td className="num">{pct(x.pred)}</td>
                <td className="num">{x.actual === null ? "—" : pct(x.actual)}</td>
                <td className="num">{x.actual === null ? "—" : pct(x.pred - x.actual)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function App() {
  const { path, go } = useRoute();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [index, setIndex] = useState<IndexRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    loadMeta().then(setMeta).catch((e) => setError(String(e)));
    loadIndex().then(setIndex).catch(() => undefined);
  }, []);
  const zipMatch = path.match(/^\/zip\/(\d{5})/);
  return (
    <div className="container">
      <header className="top">
        <a
          className="brand"
          href="/"
          onClick={(e) => {
            e.preventDefault();
            go("/");
          }}
        >
          <b>HomeSignal</b>
          <span>ZIP-level home value growth · educational</span>
        </a>
        <nav>
          <a href={`${REPO}/blob/main/reports/evaluation.md`} target="_blank" rel="noreferrer">
            Evaluation
          </a>
          <a href={`${REPO}/blob/main/reports/model_card.md`} target="_blank" rel="noreferrer">
            Model card
          </a>
          <a href={REPO} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
      </header>
      <Disclaimer />
      {error && <p className="note">Failed to load model data: {error}</p>}
      {!meta && !error && <p className="muted">Loading…</p>}
      {meta && (zipMatch ? <ZipPage zip={zipMatch[1]} meta={meta} go={go} /> : <Home meta={meta} index={index} go={go} />)}
      <footer>
        Data: Zillow Research (ZHVI, ZORI) · U.S. Census Bureau ACS 5-year · Freddie Mac PMMS · U.S. Bureau of Labor
        Statistics. Model {meta?.model ?? ""} trained on origins {meta?.training_origins[0].slice(0, 7)} →{" "}
        {meta?.training_origins[1].slice(0, 7)}. Source code and full reports on{" "}
        <a href={REPO} target="_blank" rel="noreferrer">
          GitHub
        </a>
        .
      </footer>
    </div>
  );
}
