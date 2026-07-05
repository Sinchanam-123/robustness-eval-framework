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

export default function App() {
  const [models, setModels] = useState([]);
  const [perturbations, setPerturbations] = useState([]);
  const [results, setResults] = useState([]);
  const [robustness, setRobustness] = useState([]);
  const [insights, setInsights] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedPerturbation, setSelectedPerturbation] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState("accuracy");
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
        setSelectedModel(m.models[0]?.model ?? null);
        setSelectedPerturbation(p.perturbations[0]?.name ?? null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p style={{ padding: 24 }}>Loading…</p>;
  if (error)
    return (
      <p style={{ padding: 24, color: "crimson" }}>
        Failed to load data: {error}. Is the backend running on :8000?
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
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 24, maxWidth: 1000 }}>
      <h1>Robustness Evaluation Framework</h1>
      <p style={{ color: "#555" }}>
        Adult Census Income — models stress-tested under parameterized data
        perturbations, 5-seed-averaged.
      </p>

      <section style={{ marginBottom: 16 }}>
        <label>
          Perturbation:{" "}
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
        <label style={{ marginLeft: 16 }}>
          Metric:{" "}
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
      </section>

      {currentPerturbation && (
        <p style={{ color: "#555", fontStyle: "italic" }}>
          {currentPerturbation.description}
        </p>
      )}

      <h2>Degradation curve — {selectedPerturbation}</h2>
      <p style={{ color: "#777", fontSize: 14, marginTop: 0 }}>
        Each line is one model's {selectedMetric} as perturbation severity
        increases (severity 0 = clean baseline). Error bars are ±1 std across
        the 5 seeds. A steeper drop means less robust.
      </p>
      <DegradationChart
        results={results}
        models={models}
        perturbation={selectedPerturbation}
        metric={selectedMetric}
      />

      <h2>Robustness index</h2>
      <RobustnessCards robustness={robustness} />

      <h2 style={{ marginTop: 32 }}>
        Aggregated results — {selectedPerturbation}
      </h2>
      <section style={{ marginBottom: 8 }}>
        <label>
          Model:{" "}
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
      </section>
      <table border="1" cellPadding="6" style={{ borderCollapse: "collapse" }}>
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
            <tr key={r.severity}>
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

      <h2>Insights</h2>
      <ul>
        {insights.map((line, i) => (
          <li key={i}>{line}</li>
        ))}
      </ul>
    </div>
  );
}
