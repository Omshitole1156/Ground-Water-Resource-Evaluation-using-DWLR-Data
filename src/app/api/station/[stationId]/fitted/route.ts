// src/app/api/station/[stationId]/fitted/route.ts
//
// Proxies to your Flask /api/station/<id>/fitted endpoint.
// This keeps your Flask URL server-side (never exposed to the browser).

import { NextRequest, NextResponse } from "next/server";

const FLASK_BASE = process.env.FLASK_API_URL ?? "http://localhost:5000";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ stationId: string }> }
) {
  const { stationId } = await params;
  const body = await req.json().catch(() => ({}));

  if (!stationId) {
    return NextResponse.json({ error: "stationId is required" }, { status: 400 });
  }

  try {
    const flaskRes = await fetch(
      `${FLASK_BASE}/api/station/${stationId}/fitted`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        // abort after 60 s (model training can take a moment)
        signal: AbortSignal.timeout(60_000),
      }
    );

    const data = await flaskRes.json();

    if (!flaskRes.ok) {
      return NextResponse.json(data, { status: flaskRes.status });
    }

    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "unknown error";
    return NextResponse.json(
      { error: "Failed to reach Python model server", detail: message },
      { status: 502 }
    );
  }
}