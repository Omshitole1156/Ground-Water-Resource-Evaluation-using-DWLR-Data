"use client";

import { suggestConservationStrategies } from "@/ai/flows/suggest-conservation-strategies";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Droplets,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  BrainCircuit,
} from "lucide-react";
import { useState, useMemo, useEffect, useCallback } from "react";
import { MetricCard } from "./metric-card";
import { StationChart } from "./station-chart";
import { type Station, type TimeSeriesData } from "@/lib/types";
import { loadStation } from "@/lib/data";
import { useToast } from "@/hooks/use-toast";
import { Separator } from "./ui/separator";

interface FarmerViewProps {
  stations: Station[];
}

export default function FarmerView({ stations }: FarmerViewProps) {
  const [selectedStationId, setSelectedStationId] = useState<string | undefined>(
    stations[0]?.id
  );

  // Full station data (with currentLevel + timeSeries) loaded on demand
  const [loadedStation, setLoadedStation] = useState<Station | null>(null);
  const [isLoadingStation, setIsLoadingStation] = useState(false);

  const [isLoadingStrategies, setIsLoadingStrategies] = useState(false);
  const [strategies, setStrategies] = useState<{
    strategies: string[];
    rationale: string;
  } | null>(null);
  const { toast } = useToast();

  // The shell station from the list (has name/district but currentLevel: 0)
  const shellStation = useMemo(
    () => stations.find((s) => s.id === selectedStationId),
    [stations, selectedStationId]
  );

  // Merge: use loadedStation when available, fall back to shell
  const selectedStation = loadedStation ?? shellStation;

  // Fetch full station data whenever selection changes
  const fetchStationData = useCallback(async (stationName: string) => {
    setIsLoadingStation(true);
    setLoadedStation(null);
    try {
      const full = await loadStation(stationName);
      if (full) setLoadedStation(full);
    } catch (err) {
      console.error("Failed to load station data:", err);
      toast({
        variant: "destructive",
        title: "Failed to load station",
        description: "Could not fetch data for this station.",
      });
    } finally {
      setIsLoadingStation(false);
    }
  }, [toast]);

  // Load on mount for first station, and on every selection change
  useEffect(() => {
    if (shellStation?.name) {
      fetchStationData(shellStation.name);
    }
  }, [shellStation?.name, fetchStationData]);

  const handleStationChange = (stationId: string) => {
    setSelectedStationId(stationId);
    setStrategies(null);
  };

  const handleGetStrategies = async () => {
    if (!selectedStation) return;
    setIsLoadingStrategies(true);
    setStrategies(null);
    try {
      const last30Days = (selectedStation.timeSeries ?? []).slice(-30);
      const result = await suggestConservationStrategies({
        currentGroundwaterLevel: selectedStation.currentLevel,
        predictedGroundwaterLevel: selectedStation.currentLevel,
        location: selectedStation.district,
        landUse: selectedStation.landUse,
        historicalData: JSON.stringify(last30Days),
      });
      setStrategies(result);
    } catch (error) {
      console.error("Failed to get strategies:", error);
      toast({
        variant: "destructive",
        title: "Failed to Get Strategies",
        description: "Could not generate conservation strategies.",
      });
    } finally {
      setIsLoadingStrategies(false);
    }
  };

  // Format the current level for display — show loading state while fetching
  const currentLevelDisplay = isLoadingStation
    ? "—"
    : loadedStation?.currentLevel != null
      ? `${loadedStation.currentLevel} m`
      : "N/A";

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Select Monitoring Station</CardTitle>
          <CardDescription>
            Choose a station to view detailed data and get AI-powered insights.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Select onValueChange={handleStationChange} value={selectedStationId}>
            <SelectTrigger className="w-full md:w-[300px]">
              <SelectValue placeholder="Select a station" />
            </SelectTrigger>
            <SelectContent>
              {stations.map((station) => (
                <SelectItem key={station.id} value={station.id}>
                  {station.name} - ({station.district})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      {selectedStation && (
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-6">
            <div className="grid gap-4 md:grid-cols-2">

              {/* Current Level — shows spinner while loading, real value after */}
              <MetricCard
                title="Current Level"
                value={
                  isLoadingStation ? (
                    <span className="flex items-center gap-2 text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading…
                    </span>
                  ) : currentLevelDisplay
                }
                icon={<Droplets />}
                status={loadedStation?.status ?? selectedStation.status}
              />

              <MetricCard
                title="Status"
                value={isLoadingStation ? "—" : (loadedStation?.status ?? selectedStation.status)}
                icon={
                  (loadedStation?.status ?? selectedStation.status) === "Critical" ? (
                    <ShieldAlert className="text-destructive" />
                  ) : (loadedStation?.status ?? selectedStation.status) === "Warning" ? (
                    <ShieldAlert className="text-yellow-500" />
                  ) : (
                    <ShieldCheck className="text-green-500" />
                  )
                }
                status={loadedStation?.status ?? selectedStation.status}
              />
            </div>

            <StationChart
              data={loadedStation?.timeSeries ?? selectedStation.timeSeries ?? []}
              title="Groundwater Level Trend"
              description={`Historical and predicted levels for ${selectedStation.name}.`}
              stationId={selectedStation.name}
            />
          </div>

          <div className="space-y-6">
            {/* LSTM Info Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BrainCircuit className="h-5 w-5 text-primary" />
                  Predictive Analytics
                </CardTitle>
                <CardDescription>
                  LSTM-powered groundwater forecasting.
                </CardDescription>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground space-y-2">
                <p>
                  Forecasts are generated automatically by an adaptive LSTM model
                  trained on historical observations for this station.
                </p>
                <p>
                  Use the{" "}
                  <span className="font-medium text-foreground">Forecast horizon</span>{" "}
                  buttons on the chart to switch between 7, 14, and 30-day forecasts.
                </p>
              </CardContent>
            </Card>

            {/* AI Recommendations */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  AI Recommendations
                </CardTitle>
                <CardDescription>
                  Get AI-powered conservation strategies.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingStrategies ? (
                  <div className="flex items-center justify-center p-8">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : strategies ? (
                  <div className="space-y-4 text-sm">
                    <div>
                      <h4 className="font-semibold mb-2">Suggested Strategies:</h4>
                      <ul className="list-disc pl-5 space-y-1">
                        {strategies.strategies.map((s, i) => (
                          <li key={i}>{s}</li>
                        ))}
                      </ul>
                    </div>
                    <Separator />
                    <div>
                      <h4 className="font-semibold mb-2">Rationale:</h4>
                      <p className="text-muted-foreground">{strategies.rationale}</p>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Click the button below to generate strategies based on current data.
                  </p>
                )}
              </CardContent>
              <CardFooter>
                <Button
                  onClick={handleGetStrategies}
                  disabled={isLoadingStrategies || isLoadingStation}
                >
                  {isLoadingStrategies && (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  Get Strategies
                </Button>
              </CardFooter>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
