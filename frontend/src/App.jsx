import { useEffect, useState } from "react";
import {
  fetchModels,
  fetchPerturbations,
  fetchResults,
  fetchRobustnessIndex,
  fetchInsights,
} from "./api";
import { METRICS } from "./theme";
import DegradationChart from "./components/DegradationChart";
import RobustnessCards from "./components/RobustnessCards";
import SeveritySlider from "./components/SeveritySlider";
import ContextualInsight from "./components/ContextualInsight";

export default function App() {
  const [models, setModels] = useState([]);
  const [perturbations, setPerturbations] = useState([]);
  const [results, setResults] = useState([]);
  const [robustness, setRobustness] = useState([]);
  const [insights, setInsights] = useState([]);
  const [testedSeverities, setTestedSeverities] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedPerturbation, setSelectedPerturbation] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState("accuracy");
  const [selectedSeverity, setSelectedSeverity] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetchModels(),
      fetchPerturbations(),
      fetchResults(),
      fetchRobustnessIndex(),
      fetchInsights(),
    ])
      .then(([m, p, r, ri, ins]) => {
        setModels(m.models);
        setPerturbations(p.perturbations);
        setResults(r.results);
        setRobustness(ri.robustness_index);
        setInsights(ins.insights);
        setTestedSeverities(r.tested_severities);
        setSelectedModel(m.models[0]?.model ?? null);
        setSelectedPerturbation(p.perturbations[0]?.name ?? null);
        setSelectedSeverity(r.tested_severities[0] ?? null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="state-msg">Loading robustness data…</p>;
  if (error)
    return (
      <p className="state-msg state-error">
        Failed to load data: {error}. Is the backend running on :8000? Start it
        with <code>uvicorn backend.app:app --reload</code>.
      </p>
    );
  if (results.length === 0)
    return (
      <p className="state-msg">
        No results found. Run the offline pipeline first:{" "}
        <code>python main.py</code>.
      </p>
    );

  const rows = results
    .filter(
      (r) =>
        r.model === selectedModel && r.perturbation === selectedPerturbation
    )
    .sort((a, b) => a.severity - b.severity);

  const currentPerturbation = perturbations.find(
    (p) => p.name === selectedPerturbation
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>Robustness Evaluation Framework</h1>
        <p>
          Adult Census Income — three models stress-tested under parameterized
          data perturbations, averaged across 5 seeds. Every number on this page
          traces to a row in <code>results.csv</code>.
        </p>
      </header>

      <section className="panel">
        <div className="controls">
          <label>
            Model
            <select
              value={selectedModel ?? ""}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.model} value={m.model}>
                  {m.model}
                </option>
              ))}
            </select>
          </label>
          <label>
            Perturbation
            <select
              value={selectedPerturbation ?? ""}
              onChange={(e) => setSelectedPerturbation(e.target.value)}
            >
              {perturbations.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Metric
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
            >
              {METRICS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <section className="panel">
        <h2>Degradation curve — {selectedPerturbation}</h2>
        <div className="callout">
          <strong>How to read this chart:</strong> the x-axis is perturbation
          severity (0 = clean, untouched data). The y-axis is {selectedMetric}.
          Each line is one model; a line that stays flat is robust, a line that
          drops steeply degrades under stress. Error bars show ±1 std across the
          5 seeds. The dashed marker is the severity selected below.
        </div>
        {currentPerturbation && (
          <p className="perturbation-desc">{currentPerturbation.description}</p>
        )}
        <DegradationChart
          results={results}
          models={models}
          perturbation={selectedPerturbation}
          metric={selectedMetric}
          selectedSeverity={selectedSeverity}
        />
        <SeveritySlider
          severities={testedSeverities}
          selectedSeverity={selectedSeverity}
          onChange={setSelectedSeverity}
          results={results}
          models={models}
          perturbation={selectedPerturbation}
          metric={selectedMetric}
        />
      </section>

      <section className="panel insight-panel">
        <h2>Insight</h2>
        <ContextualInsight
          results={results}
          models={models}
          model={selectedModel}
          perturbation={selectedPerturbation}
          severity={selectedSeverity}
          metric={selectedMetric}
          insights={insights}
        />
      </section>

      <section className="panel">
        <h2>Robustness index</h2>
        <p className="perturbation-desc">
          retention(severity) = metric ÷ clean-baseline metric; the index is the
          mean retention across tested severities, averaged over perturbation
          types. Higher = more robust. Cards are ordered most-robust first.
        </p>
        <RobustnessCards robustness={robustness} />
      </section>

      <section className="panel">
        <h2>
          Aggregated results — {selectedModel} / {selectedPerturbation}
        </h2>
        {rows.length === 0 ? (
          <p className="perturbation-desc">
            No aggregated rows for this selection.
          </p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>severity</th>
                {METRICS.map((m) => (
                  <th key={m}>{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.severity}
                  className={
                    r.severity === selectedSeverity ? "row-selected" : undefined
                  }
                >
                  <td>{r.severity}</td>
                  {METRICS.map((m) => (
                    <td key={m}>
                      {r[`${m}_mean`].toFixed(3)} ± {r[`${m}_std`].toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2>All generated insights</h2>
        <ul>
          {insights.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
