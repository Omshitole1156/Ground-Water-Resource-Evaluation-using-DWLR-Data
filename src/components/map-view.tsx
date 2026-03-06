'use client';

import { useState, useMemo, useCallback, useRef } from 'react';
import { Map, Marker } from 'pigeon-maps';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { Station } from '@/lib/types';
import { cn } from '@/lib/utils';
import { Droplets, MapPin, TrendingUp, TrendingDown, X, AlertTriangle, Layers, Activity, ChevronRight } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { StationChart } from './station-chart';

interface MapViewProps {
  stations: Station[];
  center: [number, number];
  zoom: number;
  title: string;
  showDepthCategory?: boolean;
}

// ── Constants ────────────────────────────────────────────────────────────────
const STATUS_COLORS: Record<Station['status'], string> = {
  Normal:   '#22c55e',
  Warning:  '#f59e0b',
  Critical: '#ef4444',
};

const STATUS_GLOW: Record<Station['status'], string> = {
  Normal:   '0 0 8px rgba(34,197,94,0.6)',
  Warning:  '0 0 8px rgba(245,158,11,0.6)',
  Critical: '0 0 10px rgba(239,68,68,0.8)',
};

const DEPTH_COLOR = (level: number): string => {
  if (level <= 2)  return '#93c5fd';
  if (level <= 5)  return '#3b82f6';
  if (level <= 10) return '#1d4ed8';
  return '#1e3a8a';
};

// ── Cluster helpers ───────────────────────────────────────────────────────────
function latLngToPixel(lat: number, lng: number, zoom: number): [number, number] {
  const x = ((lng + 180) / 360) * Math.pow(2, zoom) * 256;
  const latRad = (lat * Math.PI) / 180;
  const y = ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * Math.pow(2, zoom) * 256;
  return [x, y];
}

interface Cluster {
  id: string;
  lat: number;
  lng: number;
  stations: Station[];
  dominantStatus: Station['status'];
}

function clusterStations(stations: Station[], zoom: number, threshold = 40): Cluster[] {
  const pixels = stations.map(s => latLngToPixel(s.lat, s.lng, zoom));
  const assigned = new Array(stations.length).fill(false);
  const clusters: Cluster[] = [];

  for (let i = 0; i < stations.length; i++) {
    if (assigned[i]) continue;
    const group: number[] = [i];
    assigned[i] = true;
    for (let j = i + 1; j < stations.length; j++) {
      if (assigned[j]) continue;
      const dx = pixels[i][0] - pixels[j][0];
      const dy = pixels[i][1] - pixels[j][1];
      if (Math.sqrt(dx * dx + dy * dy) < threshold) {
        group.push(j);
        assigned[j] = true;
      }
    }
    const members = group.map(idx => stations[idx]);
    const avgLat = members.reduce((s, m) => s + m.lat, 0) / members.length;
    const avgLng = members.reduce((s, m) => s + m.lng, 0) / members.length;
    // Priority: Critical > Warning > Normal
    const dominantStatus: Station['status'] =
      members.some(m => m.status === 'Critical') ? 'Critical' :
      members.some(m => m.status === 'Warning')  ? 'Warning'  : 'Normal';

    clusters.push({
      id: `cluster-${i}`,
      lat: avgLat,
      lng: avgLng,
      stations: members,
      dominantStatus,
    });
  }
  return clusters;
}

// ── Sub-components ────────────────────────────────────────────────────────────
function StatPill({ icon, label, value, accent }: { icon: React.ReactNode; label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-center gap-2 rounded-xl bg-white/5 border border-white/10 px-3 py-2.5 backdrop-blur-sm">
      <span className="shrink-0" style={{ color: accent ?? '#60a5fa' }}>{icon}</span>
      <div className="min-w-0">
        <p className="text-[10px] font-medium uppercase tracking-widest text-white/40">{label}</p>
        <p className="text-sm font-semibold text-white truncate">{value}</p>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function MapView({
  stations,
  center,
  zoom,
  title,
  showDepthCategory = false,
}: MapViewProps) {
  const [selectedStation, setSelectedStation] = useState<Station | null>(null);
  const [currentZoom, setCurrentZoom] = useState(zoom);
  const [activeFilter, setActiveFilter] = useState<Station['status'] | 'All'>('All');
  const [showLegend, setShowLegend] = useState(true);

  // Filter stations
  const filteredStations = useMemo(
    () => activeFilter === 'All' ? stations : stations.filter(s => s.status === activeFilter),
    [stations, activeFilter]
  );

  // Cluster based on zoom — disable clustering at high zoom
  const clusters = useMemo(() => {
    if (currentZoom >= 10) {
      return filteredStations.map(s => ({
        id: s.id,
        lat: s.lat,
        lng: s.lng,
        stations: [s],
        dominantStatus: s.status,
      }));
    }
    const threshold = currentZoom >= 8 ? 25 : currentZoom >= 6 ? 45 : 65;
    return clusterStations(filteredStations, currentZoom, threshold);
  }, [filteredStations, currentZoom]);

  const handleMarkerClick = useCallback((cluster: Cluster) => {
    if (cluster.stations.length === 1) {
      setSelectedStation(cluster.stations[0]);
    } else {
      // Show the most critical station in a cluster
      const priority = cluster.stations.find(s => s.status === 'Critical')
        ?? cluster.stations.find(s => s.status === 'Warning')
        ?? cluster.stations[0];
      setSelectedStation(priority);
    }
  }, []);

  const markerColor = (cluster: Cluster) => {
    if (showDepthCategory && cluster.stations.length === 1) {
      return DEPTH_COLOR(cluster.stations[0].currentLevel);
    }
    return STATUS_COLORS[cluster.dominantStatus];
  };

  // Stats for header badges
  const stats = useMemo(() => ({
    total:    stations.length,
    critical: stations.filter(s => s.status === 'Critical').length,
    warning:  stations.filter(s => s.status === 'Warning').length,
    normal:   stations.filter(s => s.status === 'Normal').length,
  }), [stations]);

  return (
    <div className="relative w-full rounded-2xl overflow-hidden border border-white/10 bg-[#0a0f1e] shadow-2xl shadow-black/50">
      {/* ── Header ── */}
      <div className="relative z-10 flex items-center justify-between px-5 py-4 border-b border-white/8 bg-gradient-to-r from-[#0d1329]/90 to-[#0a1020]/90 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-blue-500/15 border border-blue-500/30">
            <Layers className="h-4 w-4 text-blue-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-white tracking-tight">{title}</h2>
            <p className="text-[11px] text-white/40">{stats.total} monitoring stations</p>
          </div>
        </div>

        {/* Status filter pills */}
        <div className="flex items-center gap-1.5">
          {(['All', 'Normal', 'Warning', 'Critical'] as const).map(f => (
            <button
              key={f}
              onClick={() => setActiveFilter(f)}
              className={cn(
                'text-[11px] font-medium px-2.5 py-1 rounded-full border transition-all duration-200',
                activeFilter === f
                  ? f === 'All'
                    ? 'bg-blue-500/20 border-blue-500/40 text-blue-300'
                    : f === 'Critical'
                      ? 'bg-red-500/20 border-red-500/40 text-red-300'
                      : f === 'Warning'
                        ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                        : 'bg-green-500/20 border-green-500/40 text-green-300'
                  : 'bg-white/5 border-white/10 text-white/40 hover:text-white/70 hover:bg-white/8'
              )}
            >
              {f === 'All' ? `All (${stats.total})` :
               f === 'Critical' ? `⚠ ${stats.critical}` :
               f === 'Warning'  ? `~ ${stats.warning}` :
               `✓ ${stats.normal}`}
            </button>
          ))}
        </div>
      </div>

      {/* ── Map ── */}
      <div className="relative" style={{ height: '520px' }}>
        <Map
          defaultCenter={center}
          defaultZoom={zoom}
          onBoundsChanged={({ zoom: z }) => setCurrentZoom(Math.round(z))}
          attribution={false}
          metaWheelZoom
          twoFingerDrag
        >
          {clusters.map(cluster => (
            <Marker
              key={cluster.id}
              anchor={[cluster.lat, cluster.lng]}
              width={cluster.stations.length > 1 ? 38 : 28}
              color={markerColor(cluster)}
              onClick={() => handleMarkerClick(cluster)}
            >
              {cluster.stations.length > 1 && (
                <div
                  style={{
                    position: 'absolute',
                    top: -6,
                    right: -6,
                    background: markerColor(cluster),
                    borderRadius: '50%',
                    width: 16,
                    height: 16,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 9,
                    fontWeight: 700,
                    color: '#fff',
                    border: '1.5px solid rgba(0,0,0,0.4)',
                  }}
                >
                  {cluster.stations.length}
                </div>
              )}
            </Marker>
          ))}
        </Map>

        {/* ── Legend overlay ── */}
        {showLegend && (
          <div className="absolute bottom-4 left-4 z-10 rounded-xl border border-white/15 bg-[#0a0f1e]/85 backdrop-blur-md px-3 py-2.5 space-y-1.5 shadow-xl">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-white/40 mb-1">
              {showDepthCategory ? 'Depth to Water' : 'Station Status'}
            </p>
            {showDepthCategory ? (
              <>
                {[['≤ 2 m', '#93c5fd'], ['≤ 5 m', '#3b82f6'], ['≤ 10 m', '#1d4ed8'], ['> 10 m', '#1e3a8a']].map(([label, color]) => (
                  <div key={label} className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: color }} />
                    <span className="text-[11px] text-white/70">{label}</span>
                  </div>
                ))}
              </>
            ) : (
              <>
                {Object.entries(STATUS_COLORS).map(([status, color]) => (
                  <div key={status} className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: color }} />
                    <span className="text-[11px] text-white/70">{status}</span>
                  </div>
                ))}
              </>
            )}
          </div>
        )}

        {/* ── Cluster count badge ── */}
        <div className="absolute bottom-4 right-4 z-10 rounded-xl border border-white/10 bg-[#0a0f1e]/80 backdrop-blur-md px-3 py-1.5">
          <span className="text-[11px] text-white/50">
            Showing <span className="text-white font-semibold">{clusters.length}</span> markers · zoom {currentZoom}
          </span>
        </div>
      </div>

      {/* ── Station Detail Panel ── */}
      {selectedStation && (
        <div
          className="absolute inset-0 z-20 flex items-center justify-center p-4"
          style={{ background: 'rgba(5,8,20,0.75)', backdropFilter: 'blur(6px)' }}
          onClick={e => { if (e.target === e.currentTarget) setSelectedStation(null); }}
        >
          <div
            className="w-full max-w-xl rounded-2xl border border-white/12 overflow-hidden shadow-2xl"
            style={{ background: 'linear-gradient(145deg, #0d1630 0%, #0a1020 100%)' }}
          >
            {/* Panel header */}
            <div className="flex items-start justify-between px-5 pt-5 pb-4 border-b border-white/8">
              <div className="flex items-center gap-3">
                <div
                  className="w-9 h-9 rounded-xl flex items-center justify-center border"
                  style={{
                    background: `${STATUS_COLORS[selectedStation.status]}18`,
                    borderColor: `${STATUS_COLORS[selectedStation.status]}40`,
                    boxShadow: STATUS_GLOW[selectedStation.status],
                  }}
                >
                  <Activity className="h-4 w-4" style={{ color: STATUS_COLORS[selectedStation.status] }} />
                </div>
                <div>
                  <h3 className="text-base font-semibold text-white leading-tight">{selectedStation.name}</h3>
                  <p className="text-[12px] text-white/40 flex items-center gap-1 mt-0.5">
                    <MapPin className="h-3 w-3" />
                    {selectedStation.district}, {selectedStation.state}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className="text-[11px] font-semibold px-2.5 py-1 rounded-full border"
                  style={{
                    background: `${STATUS_COLORS[selectedStation.status]}18`,
                    borderColor: `${STATUS_COLORS[selectedStation.status]}40`,
                    color: STATUS_COLORS[selectedStation.status],
                  }}
                >
                  {selectedStation.status}
                </span>
                <button
                  onClick={() => setSelectedStation(null)}
                  className="w-7 h-7 rounded-lg flex items-center justify-center bg-white/5 hover:bg-white/10 border border-white/10 text-white/50 hover:text-white transition-all"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>

            {/* Panel body */}
            <div className="p-5 space-y-4">
              {/* Stats grid */}
              <div className="grid grid-cols-3 gap-2.5">
                <StatPill
                  icon={<Droplets className="h-4 w-4" />}
                  label="Water Level"
                  value={`${selectedStation.currentLevel} m`}
                  accent="#60a5fa"
                />
                <StatPill
                  icon={<TrendingUp className="h-4 w-4" />}
                  label="Land Use"
                  value={selectedStation.landUse}
                  accent="#a78bfa"
                />
                <StatPill
                  icon={<MapPin className="h-4 w-4" />}
                  label="Station ID"
                  value={selectedStation.id}
                  accent="#34d399"
                />
              </div>

              {/* Chart */}
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-widest text-white/30 mb-2">
                  90-Day Water Level Trend
                </p>
                <div
                  className="rounded-xl border border-white/8 overflow-hidden"
                  style={{ height: 180, background: 'rgba(255,255,255,0.02)' }}
                >
                  <StationChart
                    data={selectedStation.timeSeries.slice(-90)}
                    title=""
                    description=""
                    stationId={selectedStation.name}  
                  />
                </div>
              </div>

              {/* Coordinates */}
              <div className="flex items-center justify-between text-[11px] text-white/25 pt-1 border-t border-white/6">
                <span>{selectedStation.lat.toFixed(4)}°N, {selectedStation.lng.toFixed(4)}°E</span>
                <span className="flex items-center gap-1 text-blue-400/60 hover:text-blue-400 cursor-pointer transition-colors">
                  View full report <ChevronRight className="h-3 w-3" />
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}