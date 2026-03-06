'use client';

import { useEffect, useState, useCallback, useMemo } from 'react';
import {
  Line,
  LineChart,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import type { TimeSeriesData } from '@/lib/types';

interface FittedRow {
  date: string;
  actual: number | null;
  predicted: number | null;
}

interface ModelMetrics {
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  mape: number | null;
  accuracy_score: number | null;
  tolerance_m: number;
  n_test: number;
}

interface FittedResponse {
  algo: string;
  tier: string;
  real_obs: number;
  history: FittedRow[];
  forecast: FittedRow[];
  metrics: ModelMetrics;
}

interface StationChartProps {
  data: TimeSeriesData[];
  title: string;
  description: string;
  stationId: string;
}

const chartConfig = {
  actual: {
    label: 'Actual Level (m)',
    color: 'hsl(var(--chart-1))',
  },
  predicted: {
    label: 'Predicted / Forecast (m)',
    color: 'hsl(var(--chart-2))',
  },
} satisfies ChartConfig;

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Compute Y-axis domain from only finite numeric values in the chart data.
 * Recharts' built-in 'dataMin'/'dataMax' strings include null/undefined rows,
 * which causes the axis to blow out to sentinel values like 99999.
 */
function safeYDomain(rows: FittedRow[], pad = 0.5): [number, number] {
  const vals: number[] = [];
  for (const r of rows) {
    if (r.actual   != null && isFinite(r.actual))   vals.push(r.actual);
    if (r.predicted != null && isFinite(r.predicted)) vals.push(r.predicted);
  }
  if (vals.length === 0) return [0, 10];
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  // Extra guard: if range is absurd (sentinel leaked through), fall back
  if (max - min > 500) return [0, 10];
  return [
    Math.floor((min - pad) * 10) / 10,
    Math.ceil((max + pad)  * 10) / 10,
  ];
}

function buildResidualMatrix(rows: FittedRow[]) {
  const valid = rows.filter((r) => r.actual != null && r.predicted != null);
  const buckets = [
    { label: '≤0.1 m',    max: 0.1,      over: 0, under: 0 },
    { label: '0.1–0.3 m', max: 0.3,      over: 0, under: 0 },
    { label: '0.3–0.5 m', max: 0.5,      over: 0, under: 0 },
    { label: '0.5–1 m',   max: 1.0,      over: 0, under: 0 },
    { label: '>1 m',      max: Infinity,  over: 0, under: 0 },
  ];
  for (const r of valid) {
    const err = Math.abs(r.actual! - r.predicted!);
    const isOver = r.predicted! > r.actual!;
    const b = buckets.find((b) => err <= b.max)!;
    isOver ? b.over++ : b.under++;
  }
  return buckets.map((b) => ({
    ...b,
    total: b.over + b.under,
    pct: valid.length
      ? (((b.over + b.under) / valid.length) * 100).toFixed(1)
      : '0',
  }));
}

function scoreColor(score: number | null) {
  if (score == null) return 'text-muted-foreground';
  if (score >= 80)   return 'text-emerald-600 dark:text-emerald-400';
  if (score >= 60)   return 'text-amber-500';
  return 'text-red-500';
}

function fmt(v: number | null | undefined, d = 3) {
  return v == null ? 'N/A' : v.toFixed(d);
}

// ── Component ─────────────────────────────────────────────────────────────────

export function StationChart({ data, title, description, stationId }: StationChartProps) {
  const [horizon, setHorizon] = useState(14);
  const [fitted, setFitted]   = useState<FittedResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const fetchFitted = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/station/${stationId}/fitted`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ horizon }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error ?? `HTTP ${res.status}`);
      }
      const json: FittedResponse = await res.json();
      setFitted(json);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [stationId, horizon]);

  useEffect(() => {
    fetchFitted();
  }, [fetchFitted]);

  // Build chart rows — filter out any row where BOTH values are null/non-finite
  const chartData: FittedRow[] = useMemo(() => {
    const raw: FittedRow[] = fitted
      ? [...fitted.history, ...fitted.forecast]
      : data.map((d) => ({ date: d.date, actual: d.level ?? null, predicted: null }));

    // Sanitise: replace non-finite numbers with null so Recharts ignores them
    return raw.map((r) => ({
      date:      r.date,
      actual:    r.actual    != null && isFinite(r.actual)    ? r.actual    : null,
      predicted: r.predicted != null && isFinite(r.predicted) ? r.predicted : null,
    }));
  }, [fitted, data]);

  // Safe domain computed from only valid numeric values
  const yDomain = useMemo(() => safeYDomain(chartData), [chartData]);

  const forecastStartDate = fitted?.forecast[0]?.date ?? null;
  const metrics           = fitted?.metrics ?? null;
  const matrix            = fitted ? buildResidualMatrix(fitted.history) : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">

        {/* ── Controls ── */}
        <div className="flex flex-wrap items-center gap-3">

          {/* LSTM tier badge */}
          <div className="flex items-center gap-2 rounded-lg border px-3 py-1.5 bg-primary/5">
            <span className="text-xs font-semibold text-primary uppercase tracking-wide">
              LSTM
            </span>
            {fitted?.tier && (
              <span className="text-xs text-muted-foreground capitalize">
                ({fitted.tier} · {fitted.real_obs} obs)
              </span>
            )}
          </div>

          {/* Forecast horizon */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Forecast horizon:</span>
            {[7, 14, 30].map((d) => (
              <button
                key={d}
                onClick={() => setHorizon(d)}
                disabled={loading}
                className={`rounded-md border px-2 py-1 transition-colors disabled:opacity-50
                  ${horizon === d
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'hover:bg-muted'
                  }`}
              >
                {d}d
              </button>
            ))}
          </div>

          <button
            onClick={fetchFitted}
            disabled={loading}
            className="ml-auto rounded-md border px-3 py-1 text-xs hover:bg-muted disabled:opacity-50"
          >
            {loading ? 'Running model…' : 'Re-run'}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
            ⚠ {error}
          </div>
        )}

        {/* ── Accuracy Score ── */}
        <div className="flex items-center gap-4 rounded-xl border p-4">
          <div className="flex-1">
            <p className="text-sm font-medium">Model Accuracy Score</p>
            <p className="text-xs text-muted-foreground">
              Predictions within ±{metrics?.tolerance_m ?? 0.5} m &nbsp;·&nbsp;
              {metrics ? `${metrics.n_test} test samples` : '—'}
              &nbsp;·&nbsp;
              <span className="font-medium">LSTM model</span>
            </p>
          </div>
          {loading ? (
            <div className="h-10 w-24 animate-pulse rounded-lg bg-muted" />
          ) : (
            <span className={`text-4xl font-bold tabular-nums ${scoreColor(metrics?.accuracy_score ?? null)}`}>
              {metrics?.accuracy_score != null ? `${metrics.accuracy_score.toFixed(1)}%` : 'N/A'}
            </span>
          )}
        </div>

        {/* ── Metric tiles ── */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: 'MAE',  value: fmt(metrics?.mae),     unit: 'm' },
            { label: 'RMSE', value: fmt(metrics?.rmse),    unit: 'm' },
            { label: 'R²',   value: fmt(metrics?.r2, 4),   unit: ''  },
            { label: 'MAPE', value: fmt(metrics?.mape, 2), unit: '%' },
          ].map(({ label, value, unit }) => (
            <div key={label} className="rounded-lg border px-3 py-2 text-center">
              <p className="text-xs text-muted-foreground">{label}</p>
              {loading ? (
                <div className="mx-auto mt-1 h-5 w-16 animate-pulse rounded bg-muted" />
              ) : (
                <p className="font-semibold tabular-nums">
                  {value}
                  {value !== 'N/A' && unit && (
                    <span className="ml-0.5 text-xs font-normal text-muted-foreground">{unit}</span>
                  )}
                </p>
              )}
            </div>
          ))}
        </div>

        {/* ── Accuracy Matrix ── */}
        {matrix.length > 0 && (
          <div>
            <p className="mb-2 text-sm font-medium">Accuracy Matrix — Error Distribution</p>
            <div className="overflow-x-auto rounded-lg border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50 text-xs text-muted-foreground">
                    <th className="px-3 py-2 text-left">Error Bucket</th>
                    <th className="px-3 py-2 text-right">Over-predicted</th>
                    <th className="px-3 py-2 text-right">Under-predicted</th>
                    <th className="px-3 py-2 text-right">Total</th>
                    <th className="px-3 py-2 text-right">% of samples</th>
                    <th className="px-3 py-2 text-left">Distribution</th>
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((row, i) => (
                    <tr
                      key={row.label}
                      className={`border-b last:border-0 ${i % 2 !== 0 ? 'bg-muted/30' : ''}`}
                    >
                      <td className="px-3 py-2 font-medium">{row.label}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-blue-500">{row.over}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-orange-500">{row.under}</td>
                      <td className="px-3 py-2 text-right tabular-nums font-semibold">{row.total}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{row.pct}%</td>
                      <td className="px-3 py-2">
                        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${row.pct}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className="bg-muted/50 text-xs text-muted-foreground">
                    <td className="px-3 py-2 font-semibold">Total</td>
                    <td className="px-3 py-2 text-right tabular-nums text-blue-500 font-semibold">
                      {matrix.reduce((s, r) => s + r.over, 0)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-orange-500 font-semibold">
                      {matrix.reduce((s, r) => s + r.under, 0)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums font-semibold">
                      {matrix.reduce((s, r) => s + r.total, 0)}
                    </td>
                    <td className="px-3 py-2 text-right">100%</td>
                    <td />
                  </tr>
                </tfoot>
              </table>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              <span className="text-blue-500">Blue = over-predicted</span>
              {' · '}
              <span className="text-orange-500">Orange = under-predicted</span>
            </p>
          </div>
        )}

        {/* ── Line chart ── */}
        <ChartContainer config={chartConfig} className="h-[280px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickFormatter={(v) =>
                  new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
                }
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                // FIX: explicit numeric domain computed from only valid values.
                // The old 'dataMin - 1' string form lets Recharts include
                // null/99999 sentinel rows, blowing the axis to 99996.
                domain={yDomain}
                tickFormatter={(v) => v.toFixed(1)}
              />
              <ChartTooltip cursor={false} content={<ChartTooltipContent indicator="line" />} />
              {forecastStartDate && (
                <ReferenceLine
                  x={forecastStartDate}
                  stroke="hsl(var(--muted-foreground))"
                  strokeDasharray="4 4"
                  label={{ value: 'Forecast →', position: 'insideTopRight', fontSize: 10 }}
                />
              )}
              <Line
                dataKey="actual"
                name="actual"
                type="monotone"
                stroke="var(--color-actual)"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
              />
              <Line
                dataKey="predicted"
                name="predicted"
                type="monotone"
                stroke="var(--color-predicted)"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <p className="text-xs text-muted-foreground">
          Solid line = actual measurements · Dashed line = LSTM predictions &amp; forecast
        </p>
      </CardContent>
    </Card>
  );
}
