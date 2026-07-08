import { MODEL_COLORS } from "../theme";

// Slider snapped to the tested severity grid. It's index-based (0..N-1) rather
// than a float range so it can only ever land on a real tested severity — the
// readout below is genuine 5-seed-averaged data from /api/results, never
// interpolated. If a live single-seed preview for off-grid severities is ever
// added, it must be visibly labeled as such; this component intentionally
// exposes only the tested grid.
export default function SeveritySlider({
  severities,
  selectedSeverity,
  onChange,
  results,
  models,
  perturbation,
  metric,
}) {
  const index = severities.indexOf(selectedSeverity);

  const readout = models.map((m) => {
    const row = results.find(
      (r) =>
        r.model === m.model &&
        r.perturbation === perturbation &&
        r.severity === selectedSeverity
    );
    return {
      model: m.model,
      mean: row ? row[`${metric}_mean`] : null,
      std: row ? row[`${metric}_std`] : null,
    };
  });

  return (
    <div style={{ margin: "8px 0 16px" }}>
      <label style={{ display: "block", marginBottom: 6 }}>
        Severity: <strong>{selectedSeverity}</strong>{" "}
        <span style={{ color: "#888", fontSize: 12 }}>
          (snapped to tested grid — 5-seed-averaged)
        </span>
      </label>
      <input
        type="range"
        min={0}
        max={severities.length - 1}
        step={1}
        value={index < 0 ? 0 : index}
        onChange={(e) => onChange(severities[Number(e.target.value)])}
        style={{ width: 320 }}
        list="severity-ticks"
      />
      <datalist id="severity-ticks">
        {severities.map((s, i) => (
          <option key={s} value={i} label={String(s)} />
        ))}
      </datalist>
      <div style={{ display: "flex", justifyContent: "space-between", width: 320, fontSize: 12, color: "#666" }}>
        {severities.map((s) => (
          <span key={s}>{s}</span>
        ))}
      </div>

      <table style={{ marginTop: 12, borderCollapse: "collapse", fontSize: 14 }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", paddingRight: 16 }}>model</th>
            <th style={{ textAlign: "left" }}>
              {metric} at severity {selectedSeverity}
            </th>
          </tr>
        </thead>
        <tbody>
          {readout.map((r) => (
            <tr key={r.model}>
              <td style={{ paddingRight: 16 }}>
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    borderRadius: 2,
                    background: MODEL_COLORS[r.model],
                    marginRight: 6,
                  }}
                />
                {r.model}
              </td>
              <td>
                {r.mean != null
                  ? `${r.mean.toFixed(3)} ± ${r.std.toFixed(3)}`
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
