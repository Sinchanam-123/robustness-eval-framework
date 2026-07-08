import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
  ErrorBar,
} from "recharts";
import { MODEL_COLORS } from "../theme";

// Measure the parent width once on mount and on window resize. This keeps the
// chart responsive without recharts' ResponsiveContainer, whose ResizeObserver
// re-render loop otherwise keeps the renderer from ever going idle.
function useContainerWidth(fallback = 900) {
  const ref = useRef(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const measure = () => {
      if (ref.current) setWidth(ref.current.clientWidth);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);
  return [ref, width];
}

// Builds one data point per severity (including a severity=0 anchor from the
// clean baseline metrics), with a mean + std field per model. This mirrors
// the offline matplotlib degradation curve exactly — the curve shows the full
// decay from the unperturbed anchor, not just the perturbed grid.
function buildChartData(results, models, perturbation, metric) {
  const modelNames = models.map((m) => m.model);

  const anchor = { severity: 0 };
  models.forEach((m) => {
    anchor[m.model] = m[metric];
    anchor[`${m.model}_std`] = 0;
  });

  const bySeverity = new Map([[0, anchor]]);
  results
    .filter((r) => r.perturbation === perturbation)
    .forEach((r) => {
      if (!bySeverity.has(r.severity)) {
        bySeverity.set(r.severity, { severity: r.severity });
      }
      const point = bySeverity.get(r.severity);
      point[r.model] = r[`${metric}_mean`];
      point[`${r.model}_std`] = r[`${metric}_std`];
    });

  return {
    data: [...bySeverity.values()].sort((a, b) => a.severity - b.severity),
    modelNames,
  };
}

export default function DegradationChart({
  results,
  models,
  perturbation,
  metric,
  selectedSeverity,
}) {
  const { data, modelNames } = buildChartData(results, models, perturbation, metric);
  const [ref, width] = useContainerWidth();

  return (
    <div ref={ref} style={{ width: "100%" }}>
      <LineChart
        width={width}
        height={360}
        data={data}
        margin={{ top: 8, right: 24, bottom: 24, left: 8 }}
      >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
          <XAxis
            dataKey="severity"
            type="number"
            domain={[0, 0.5]}
            ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5]}
            label={{ value: "Severity", position: "insideBottom", offset: -12 }}
          />
          <YAxis
            domain={["auto", "auto"]}
            tickFormatter={(v) => v.toFixed(2)}
            label={{
              value: metric,
              angle: -90,
              position: "insideLeft",
              style: { textAnchor: "middle" },
            }}
          />
          <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)} />
          <Legend />
          {selectedSeverity != null && (
            <ReferenceLine
              x={selectedSeverity}
              stroke="#888"
              strokeDasharray="4 4"
              label={{ value: `severity ${selectedSeverity}`, position: "top", fontSize: 11 }}
            />
          )}
          {modelNames.map((name) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={MODEL_COLORS[name]}
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls
              isAnimationActive={false}
            >
              <ErrorBar
                dataKey={`${name}_std`}
                stroke={MODEL_COLORS[name]}
                strokeWidth={1}
                width={4}
                direction="y"
              />
            </Line>
          ))}
      </LineChart>
    </div>
  );
}
