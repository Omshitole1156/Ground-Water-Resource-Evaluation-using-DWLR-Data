/**
 * data.ts — fetches real station data from Flask backend
 */

import type { Station, TimeSeriesData } from "@/lib/types";

const isServer = typeof window === "undefined";
const API_BASE = isServer ? "http://localhost:5000" : "";

// ─── Raw API shapes ───────────────────────────────────────────────────────────

interface ApiStation {
  stationName: string;
  stationCode?: string;
  district: string;
  latitude: number;
  longitude: number;
  aquiferType?: string;
  wellDepth?: number;
  real_obs: number;
  model_tier: "deep" | "medium" | "simple";
  model_trained: boolean;
  latest_level?: number | null;
  state?: string;
  landUse?: "Urban" | "Agriculture" | "Industrial" | "Rural";
}

interface ApiTimeSeriesRow {
  date: string;
  water_level: number | null;
}

interface ApiDataResponse {
  stationName: string;
  real_obs: number;
  time_series: ApiTimeSeriesRow[];
  total_days: number;
  missing_pct: number;
  date_min: string;
  date_max: string;
}

interface ApiSummary {
  stationName: string;
  district: string;
  real_obs: number;
  model_tier: string;
  model_trained: boolean;
  date_min: string;
  date_max: string;
  latitude: number;
  longitude: number;
  aquiferType?: string;
  wellDepth?: number;
  water_level: {
    min: number;
    max: number;
    mean: number;
    latest: number | null;
  };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function deriveStatus(level: number | null): Station["status"] {
  if (level === null || level === undefined) return "Normal";
  if (level < 2) return "Critical";
  if (level < 5) return "Warning";
  return "Normal";
}

function inferLandUse(station: ApiStation): Station["landUse"] {
  if (station.landUse) return station.landUse;
  const name = station.stationName.toLowerCase();
  if (name.includes("urban") || name.includes("city")) return "Urban";
  if (name.includes("indus")) return "Industrial";
  return "Agriculture";
}

/**
 * Get the last valid (non-null, finite) water level from a time series.
 * Returns null if no valid value found — never falls back to 0.
 */
function lastValidLevel(timeSeries: TimeSeriesData[]): number | null {
  for (let i = timeSeries.length - 1; i >= 0; i--) {
    const v = timeSeries[i].level;
    if (v != null && isFinite(v)) return v;
  }
  return null;
}

// ─── Loaders ──────────────────────────────────────────────────────────────────

/**
 * LIGHTWEIGHT — used on page load.
 * Single API call, no time-series, fast.
 * currentLevel is null here — FarmerView calls loadStation() on selection
 * to get the real value.
 */
export async function loadStationList(): Promise<Station[]> {
  const res = await fetch(`${API_BASE}/api/stations`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch station list: ${res.status}`);

  const apiStations: ApiStation[] = await res.json();

  return apiStations.map((s) => {
    // Use latest_level from the list API if available (non-null, finite)
    const latest =
      s.latest_level != null && isFinite(s.latest_level) ? s.latest_level : null;

    return {
      id:           s.stationCode ?? s.stationName,
      name:         s.stationName,
      district:     s.district,
      state:        s.state ?? "Maharashtra",
      lat:          s.latitude,
      lng:          s.longitude,
      landUse:      inferLandUse(s),
      currentLevel: latest,          // null until loadStation() fills it in
      status:       deriveStatus(latest),
      timeSeries:   [],
    };
  });
}

/**
 * HEAVY — called when a user selects a station.
 * Fetches summary + time-series for ONE station.
 * This is the source of truth for currentLevel.
 */
export async function loadStation(stationName: string): Promise<Station | null> {
  try {
    const encoded = encodeURIComponent(stationName);

    const [summaryRes, dataRes] = await Promise.all([
      fetch(`${API_BASE}/api/station/${encoded}/summary`, { cache: "no-store" }),
      fetch(`${API_BASE}/api/station/${encoded}/data`,    { cache: "no-store" }),
    ]);

    if (!summaryRes.ok || !dataRes.ok) return null;

    const summary: ApiSummary     = await summaryRes.json();
    const data:    ApiDataResponse = await dataRes.json();

    const timeSeries: TimeSeriesData[] = data.time_series
      .filter((row) => row.water_level != null && isFinite(row.water_level as number))
      .map((row) => ({
        date:  row.date,
        level: row.water_level as number,
      }));

    // Priority: summary.water_level.latest → last valid in timeSeries → null
    // Never fall back to 0 — null means "unknown", 0 means "zero metres"
    const summaryLatest =
      summary.water_level?.latest != null && isFinite(summary.water_level.latest)
        ? summary.water_level.latest
        : null;

    const currentLevel = summaryLatest ?? lastValidLevel(timeSeries);

    return {
      id:           summary.stationName,
      name:         summary.stationName,
      district:     summary.district,
      state:        "Maharashtra",
      lat:          summary.latitude  ?? 0,
      lng:          summary.longitude ?? 0,
      landUse:      "Agriculture",
      currentLevel,
      status:       deriveStatus(currentLevel),
      timeSeries,
    };
  } catch {
    return null;
  }
}

/**
 * BULK — only use for pre-loading all stations with time-series.
 * NOT called on page load — too slow for 1000+ stations.
 */
export async function loadAllStations(): Promise<Station[]> {
  const listRes = await fetch(`${API_BASE}/api/stations`, { cache: "no-store" });
  if (!listRes.ok) throw new Error(`Failed to fetch station list: ${listRes.status}`);
  const apiStations: ApiStation[] = await listRes.json();

  const BATCH = 10;
  const stations: Station[] = [];

  for (let i = 0; i < apiStations.length; i += BATCH) {
    const batch = apiStations.slice(i, i + BATCH);
    const results = await Promise.allSettled(
      batch.map((s) =>
        fetch(`${API_BASE}/api/station/${encodeURIComponent(s.stationName)}/data`, {
          cache: "no-store",
        })
          .then((r) => (r.ok ? (r.json() as Promise<ApiDataResponse>) : null))
          .catch(() => null)
      )
    );

    batch.forEach((apiStation, idx) => {
      const result   = results[idx];
      const dataResp = result.status === "fulfilled" ? result.value : null;

      const timeSeries: TimeSeriesData[] = dataResp
        ? dataResp.time_series
            .filter((row) => row.water_level != null && isFinite(row.water_level as number))
            .map((row) => ({ date: row.date, level: row.water_level as number }))
        : [];

      const currentLevel = lastValidLevel(timeSeries);

      stations.push({
        id:           apiStation.stationCode ?? apiStation.stationName,
        name:         apiStation.stationName,
        district:     apiStation.district,
        state:        apiStation.state ?? "Maharashtra",
        lat:          apiStation.latitude,
        lng:          apiStation.longitude,
        landUse:      inferLandUse(apiStation),
        currentLevel,
        status:       deriveStatus(currentLevel),
        timeSeries,
      });
    });
  }

  return stations;
}