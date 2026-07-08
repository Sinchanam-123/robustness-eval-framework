import { MODEL_COLORS } from "../theme";

// Robustness index as a card set (not buried in a tooltip): overall score
// per model, with the per-perturbation breakdown underneath. Cards are sorted
// most-robust-first, matching the API's ordering.
export default function RobustnessCards({ robustness }) {
  return (
    <div className="card-grid">
      {robustness.map((entry) => (
        <div
          key={entry.model}
          className="robustness-card"
          style={{ borderTopColor: MODEL_COLORS[entry.model] }}
        >
          <div className="model-name">{entry.model}</div>
          <div className="index-value">
            {entry.overall_robustness_index.toFixed(3)}
          </div>
          <div className="index-label">overall robustness index</div>
          <table>
            <tbody>
              {Object.entries(entry.per_perturbation).map(([pert, val]) => (
                <tr key={pert}>
                  <td style={{ color: "#555" }}>{pert}</td>
                  <td style={{ textAlign: "right" }}>{val.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
