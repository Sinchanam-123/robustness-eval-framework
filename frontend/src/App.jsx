import { useEffect, useState } from "react";
import {
  fetchModels,
  fetchPerturbations,
  fetchResults,
  fetchRobustnessIndex,
  fetchInsights,
} from "./api";

const METRICS = ["accuracy", "precision", "recall", "f1"];

export default function App() {
  const [models, setModels] = useState([]);
  const [perturbations, setPerturbations] = useState([]);
  const [results, setResults] = useState([]);
  const [robustness, setRobustness] = useState([]);
  const [insights, setInsights] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedPerturbation, setSelectedPerturbation] = useState(null);
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
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 24, maxWidth: 900 }}>
      <h1>Robustness Evaluation Framework</h1>
      <p style={{ color: "#555" }}>
        Adult Census Income — models stress-tested under parameterized data
        perturbations, 5-seed-averaged.
      </p>

      <section style={{ marginBottom: 16 }}>
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
        <label style={{ marginLeft: 16 }}>
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
      </section>

      {currentPerturbation && (
        <p style={{ color: "#555", fontStyle: "italic" }}>
          {currentPerturbation.description}
        </p>
      )}

      <h2>Aggregated results (mean ± std across 5 seeds)</h2>
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

      <h2>Robustness index (overall)</h2>
      <table border="1" cellPadding="6" style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th>model</th>
            <th>overall robustness index</th>
          </tr>
        </thead>
        <tbody>
          {robustness.map((r) => (
            <tr key={r.model}>
              <td>{r.model}</td>
              <td>{r.overall_robustness_index.toFixed(3)}</td>
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
