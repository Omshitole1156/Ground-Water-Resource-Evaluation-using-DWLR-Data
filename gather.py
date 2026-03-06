#!/usr/bin/env python3
"""
gather.py

Downloads groundwater records from IndiaWRIS for Maharashtra
and outputs data DIRECTLY in the TypeScript Station format.

Saves progress after EVERY page so no data is lost if it crashes.

Outputs:
  - src/lib/realData.ts   <- drop-in replacement for mockData.ts
  - data/raw_backup.json  <- raw backup (saved progressively after every page)
"""

import requests, time, json, re, math
from pathlib import Path
import pandas as pd

# ---------- CONFIG ----------
BASE              = "https://indiawris.gov.in/Dataset/Ground%20Water%20Level"
DISTRICT_ENDPOINT = "https://indiawris.gov.in/api/v1/master/getDistrictList"
HOMEPAGE          = "https://indiawris.gov.in/groundwater"

STATE           = "Maharashtra"
AGENCY          = "CGWB"
START           = "1994-01-01"
END             = "2025-12-31"
PAGE_SIZE       = 500
REQUEST_TIMEOUT = 30

OUT_DIR  = Path("out_indiawris")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_TS = Path("src/lib/realData.ts")
BACKUP   = Path("data/raw_backup.json")
FINAL_TS.parent.mkdir(parents=True, exist_ok=True)
BACKUP.parent.mkdir(parents=True, exist_ok=True)

    # ---------- LAND USE MAP ----------
    LAND_USE_MAP = {
        "Pune":        "Urban",
        "Nagpur":      "Urban",
        "Nashik":      "Agriculture",
        "Aurangabad":  "Urban",
        "Solapur":     "Rural",
        "Kolhapur":    "Agriculture",
        "Amravati":    "Agriculture",
        "Akola":       "Agriculture",
        "Latur":       "Rural",
        "Nanded":      "Rural",
        "Chandrapur":  "Industrial",
        "Yavatmal":    "Agriculture",
        "Beed":        "Rural",
        "Jalgaon":     "Agriculture",
        "Dhule":       "Agriculture",
        "Wardha":      "Agriculture",
        "Buldhana":    "Agriculture",
        "Washim":      "Agriculture",
        "Hingoli":     "Rural",
        "Parbhani":    "Rural",
        "Osmanabad":   "Rural",
        "Gondia":      "Agriculture",
        "Bhandara":    "Agriculture",
        "Gadchiroli":  "Rural",
        "Raigad":      "Rural",
        "Ratnagiri":   "Rural",
        "Sindhudurg":  "Rural",
        "Sangli":      "Agriculture",
        "Satara":      "Agriculture",
        "Mumbai":      "Urban",
        "Thane":       "Industrial",
        "Palghar":     "Rural",
        "Ahmednagar":  "Agriculture",
        "Jalna":       "Agriculture",
        "Nandurbar":   "Rural",
    }

FALLBACK_DISTRICTS = [
    "Ahmednagar","Akola","Amravati","Aurangabad","Beed","Bhandara","Buldhana","Chandrapur",
    "Dhule","Gadchiroli","Gondia","Hingoli","Jalgaon","Jalna","Kolhapur","Latur","Mumbai",
    "Mumbai Suburban","Nagpur","Nanded","Nandurbar","Nashik","Osmanabad","Palghar","Parbhani",
    "Pune","Raigad","Ratnagiri","Sangli","Satara","Sindhudurg","Solapur","Thane","Wardha","Washim","Yavatmal"
]

# =========================================================
session = requests.Session()
session.headers.update({
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
})

# =========================================================
# HELPERS
# =========================================================

def get_status(level):
    if level < 5:  return "Critical"
    if level < 10: return "Warning"
    return "Normal"

def get_land_use(district):
    for key, val in LAND_USE_MAP.items():
        if key.lower() in district.lower():
            return val
    return "Rural"

def infer_district(station_name):
    if not station_name:
        return "Unknown"
    return str(station_name).split()[0]

def save_progress(all_rows):
    """Save raw rows to backup JSON after every page."""
    with open(BACKUP, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, default=str)

def load_progress():
    """Load previously saved rows if backup exists (resume support)."""
    if BACKUP.exists():
        print(f"Found existing backup: {BACKUP} — loading to resume...")
        with open(BACKUP, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# =========================================================
# API FETCHING
# =========================================================

def get_districts_for_state(state_name):
    headers = {
        "User-Agent": session.headers.get("User-Agent"),
        "Accept":     "application/json, text/plain, */*",
        "Referer":    HOMEPAGE,
        "Origin":     "https://indiawris.gov.in",
    }
    for method in ["GET", "POST"]:
        try:
            print(f"Trying district API ({method})...")
            if method == "GET":
                r = session.get(DISTRICT_ENDPOINT, params={"state": state_name}, headers=headers, timeout=REQUEST_TIMEOUT)
            else:
                r = session.post(DISTRICT_ENDPOINT, json={"state": state_name}, headers=headers, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            js = r.json()
            data = js.get("data") if isinstance(js, dict) else None
            if isinstance(data, list) and data:
                districts = [d.get("districtName") for d in data if d.get("districtName")]
                if districts:
                    print(f"Found {len(districts)} districts via API ({method}).")
                    return districts
        except Exception as e:
            print(f"{method} failed:", e)
    print("Using fallback district list.")
    return FALLBACK_DISTRICTS

def build_params(page=0, district=None, extra=None):
    params = {
        "page": page, "size": PAGE_SIZE, "download": "false",
        "startdate": START, "enddate": END, "stateName": STATE
    }
    if district: params["districtName"] = district
    if AGENCY:   params["agencyName"]   = AGENCY
    if extra:    params.update(extra)
    return params

def fetch_page(page, district, retries=3, backoff=2, extra_params=None):
    params = build_params(page, district, extra_params)
    for attempt in range(1, retries + 1):
        try:
            r = session.post(
                BASE, params=params,
                headers={"Referer": HOMEPAGE, "Origin": "https://indiawris.gov.in"},
                data="", timeout=REQUEST_TIMEOUT
            )
            r.raise_for_status()
            return r
        except requests.exceptions.TooManyRedirects:
            return None
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            time.sleep(backoff * attempt)
    return None

def parse_page_json(js):
    rows = []
    recs = []
    if isinstance(js, list):
        recs = js
    elif isinstance(js, dict):
        for k in ("data", "content", "result", "rows"):
            if k in js and isinstance(js[k], list):
                recs = js[k]
                break

    for rec in recs:
        try:
            sid   = rec.get("stationCode") or rec.get("stationId") or rec.get("wellNo") or rec.get("Wellno") or rec.get("stationID")
            sname = rec.get("stationName") or rec.get("village") or rec.get("siteName") or rec.get("wellName")
            lat   = rec.get("latitude")  or rec.get("Latitude")  or rec.get("lat")
            lon   = rec.get("longitude") or rec.get("Longitude") or rec.get("lon") or rec.get("long")
            dt    = rec.get("dataTime")  or rec.get("data_time") or rec.get("date") or rec.get("dataTimeStamp") or rec.get("dateTime")
            val   = rec.get("dataValue") or rec.get("data_value") or rec.get("value") or rec.get("wl") or rec.get("waterLevel")

            if (dt is None or val is None) and isinstance(rec.get("data"), list):
                for obs in rec["data"]:
                    dt2  = obs.get("datetime") or obs.get("dataTime") or obs.get("date") or obs.get("time")
                    val2 = obs.get("value") or obs.get("dataValue") or obs.get("wl")
                    if dt2 and val2 is not None:
                        rows.append({
                            "station_id":    str(sid),
                            "station_name":  sname,
                            "latitude":      float(lat) if lat not in (None, "") else None,
                            "longitude":     float(lon) if lon not in (None, "") else None,
                            "timestamp":     dt2,
                            "water_level_m": float(val2)
                        })
                continue

            if dt is None or val is None:
                continue

            rows.append({
                "station_id":    str(sid),
                "station_name":  sname,
                "latitude":      float(lat) if lat not in (None, "") else None,
                "longitude":     float(lon) if lon not in (None, "") else None,
                "timestamp":     dt,
                "water_level_m": float(val)
            })
        except Exception:
            continue
    return rows

# =========================================================
# CONVERT rows -> TypeScript file
# =========================================================

def build_typescript(all_rows):
    if not all_rows:
        print("No rows to convert.")
        return

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "water_level_m", "station_id"])
    df = df.sort_values(["station_id", "timestamp"])

    ts_stations = []

    for station_id, group in df.groupby("station_id"):
        group    = group.sort_values("timestamp")
        row0     = group.iloc[0]
        name     = str(row0.get("station_name") or station_id)
        lat      = float(row0["latitude"])  if pd.notna(row0.get("latitude"))  else 19.7515
        lng      = float(row0["longitude"]) if pd.notna(row0.get("longitude")) else 75.7139
        district = infer_district(name)
        land_use = get_land_use(district)

        time_series = []
        for _, obs in group.tail(365).iterrows():
            lvl = round(float(obs["water_level_m"]), 2)
            if math.isfinite(lvl):
                time_series.append({
                    "date":  obs["timestamp"].strftime("%Y-%m-%d"),
                    "level": lvl
                })

        if not time_series:
            continue

        current_level = time_series[-1]["level"]
        status        = get_status(current_level)

        ts_stations.append({
            "id":           str(station_id),
            "name":         name,
            "lat":          round(lat, 4),
            "lng":          round(lng, 4),
            "district":     district,
            "state":        "Maharashtra",
            "landUse":      land_use,
            "currentLevel": current_level,
            "status":       status,
            "timeSeries":   time_series,
        })

    lines = [
        "// AUTO-GENERATED by gather.py — do not edit manually",
        "// Source: IndiaWRIS DWLR groundwater data (Maharashtra 2025)",
        "",
        "import type { Station, TimeSeriesData } from './types';",
        "",
        "export const mockStationData: Station[] = [",
    ]

    for s in ts_stations:
        ts_entries = ", ".join(
            f'{{ date: "{e["date"]}", level: {e["level"]} }}'
            for e in s["timeSeries"]
        )
        line = (
            f'  {{ id: "{s["id"]}", name: "{s["name"]}", '
            f'lat: {s["lat"]}, lng: {s["lng"]}, '
            f'district: "{s["district"]}", state: "{s["state"]}", '
            f'landUse: \'{s["landUse"]}\' as const, '
            f'currentLevel: {s["currentLevel"]}, '
            f'status: \'{s["status"]}\' as const, '
            f'timeSeries: [{ts_entries}] }},'
        )
        lines.append(line)

    lines.append("];")
    lines.append("")

    FINAL_TS.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ TypeScript file written: {FINAL_TS}")
    print(f"   Total stations: {len(ts_stations)}")

# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 50)
    print("AquaWatch IndiaWRIS Data Fetcher")
    print("=" * 50)

    # Load existing progress if any (resume support)
    all_rows = load_progress()
    print(f"Starting with {len(all_rows)} rows already saved.\n")

    print("Initial GET to obtain cookies...")
    try:
        session.get(HOMEPAGE, timeout=REQUEST_TIMEOUT)
    except Exception:
        pass

    districts = get_districts_for_state(STATE)

    for district in districts:
        print(f"\n>>> District: {district}")
        page = 0
        while True:
            print(f"  Fetching page {page}...")
            r = fetch_page(page, district)
            if r is None:
                print(f"  Failed. Skipping {district}.")
                break

            try:
                js = r.json()
            except Exception:
                print("  Bad JSON. Skipping.")
                break

            rows = parse_page_json(js)
            print(f"  Page {page} -> {len(rows)} rows extracted")

            if rows:
                all_rows.extend(rows)
                # Save after EVERY page so no data is lost on crash/stop
                save_progress(all_rows)
                print(f"  Saved to backup — {len(all_rows)} total rows so far")

            # Check if more pages
            top_len = 0
            if isinstance(js, list):
                top_len = len(js)
            elif isinstance(js, dict):
                for k in ("data", "content", "result", "rows"):
                    if k in js and isinstance(js[k], list):
                        top_len = len(js[k])
                        break

            if top_len == 0 or top_len < PAGE_SIZE:
                print(f"  Last page for {district}.")
                break

            page += 1
            time.sleep(0.25)

    print(f"\n{'=' * 50}")
    print(f"Fetching complete! Total rows: {len(all_rows)}")
    print(f"Raw backup: {BACKUP}")
    print(f"Converting to TypeScript...")
    print(f"{'=' * 50}\n")

    build_typescript(all_rows)

    print(f"\nFinal step:")
    print(f"  copy src\\lib\\realData.ts src\\lib\\mockData.ts")
    print(f"  npm run dev")

if __name__ == "__main__":
    main()
