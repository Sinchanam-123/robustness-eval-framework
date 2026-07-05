import { MODEL_COLORS } from "../theme";

// Robustness index as a card set (not buried in a tooltip): overall score
// per model, with the per-perturbation breakdown underneath. Cards are sorted
// most-robust-first, matching the API's ordering.
export default function RobustnessCards({ robustness }) {
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
      {robustness.map((entry) => (
        <div
          key={entry.model}
          style={{
            border: "1px solid #ddd",
            borderTop: `4px solid ${MODEL_COLORS[entry.model]}`,
            borderRadius: 8,
            padding: 16,
            minWidth: 220,
            background: "#fff",
          }}
        >
          <div style={{ fontWeight: 600 }}>{entry.model}</div>
          <div style={{ fontSize: 32, fontWeight: 700, margin: "4px 0" }}>
            {entry.overall_robustness_index.toFixed(3)}
          </div>
          <div style={{ fontSize: 12, color: "#777", marginBottom: 8 }}>
            overall robustness index
          </div>
          <table style={{ fontSize: 13, width: "100%" }}>
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
