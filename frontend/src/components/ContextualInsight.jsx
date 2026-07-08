// Contextual insight text that updates with the model / perturbation /
// severity selection. Every statement here is derived from real fetched
// numbers (the same seed-averaged /api/results and baseline metrics the chart
// uses) — nothing templated is shown "regardless of outcome". The retention
// percentage is computed live, so if the numbers said the model collapsed,
// this text would say so too.
export default function ContextualInsight({
  results,
  models,
  model,
  perturbation,
  severity,
  metric,
  insights,
}) {
  const baselineRow = models.find((m) => m.model === model);
  const resultRow = results.find(
    (r) =>
      r.model === model &&
      r.perturbation === perturbation &&
      r.severity === severity
  );

  let computed = null;
  if (baselineRow && resultRow) {
    const baseline = baselineRow[metric];
    const current = resultRow[`${metric}_mean`];
    const retention = current / baseline;
    computed = (
      <p>
        At severity <strong>{severity}</strong> under{" "}
        <strong>{perturbation}</strong>, <strong>{model}</strong> retains{" "}
        <strong>{(retention * 100).toFixed(1)}%</strong> of its clean {metric} (
        {current.toFixed(3)} vs {baseline.toFixed(3)} baseline).
      </p>
    );
  }

  // Global generated insights that mention the selected model — surfaced
  // alongside the contextual line so the panel reflects the current selection.
  const relevant = insights.filter((line) => line.includes(model));

  return (
    <div>
      {computed}
      {relevant.length > 0 && (
        <ul style={{ marginTop: 4 }}>
          {relevant.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
