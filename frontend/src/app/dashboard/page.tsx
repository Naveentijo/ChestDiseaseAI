"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Navbar from "@/components/Navbar";
import { apiService } from "@/services/api";
import {
  FileText,
  Activity,
  Server,
  AlertCircle,
  Clock,
  ArrowUpRight,
  TrendingUp,
} from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  const [userName, setUserName] = useState("Clinician");

  useEffect(() => {
    // Session Guard
    if (localStorage.getItem("isLoggedIn") !== "true") {
      router.push("/login");
    }
  }, [router]);

  // Fetch prediction history
  const historyQuery = useQuery({
    queryKey: ["predictionHistory"],
    queryFn: () => apiService.getHistory(),
  });

  // Fetch system health
  const healthQuery = useQuery({
    queryKey: ["systemHealth"],
    queryFn: () => apiService.getHealth(),
    refetchInterval: 10000, // Poll health every 10s
  });

  const totalScans = historyQuery.data?.length || 0;
  const positiveScans = historyQuery.data?.filter(
    (item) => item.detected_diseases.length > 0
  ).length || 0;
  const healthStatus = healthQuery.data?.status || "disconnected";
  const deviceName = healthQuery.data?.device || "N/A";

  const latestScans = historyQuery.data?.slice(0, 5) || [];

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      {/* Navigation Sidebar */}
      <Navbar />

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto p-8">
        <header className="mb-8 flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-slate-100">Welcome Back</h1>
            <p className="text-slate-400 mt-1">Diagnostic triage overview and system analytics.</p>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-sm text-slate-300">
            <Clock className="h-4 w-4 text-sky-400" />
            {new Date().toLocaleDateString(undefined, {
              weekday: "long",
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </div>
        </header>

        {/* Diagnostic Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Stat 1: Total Scans */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 relative overflow-hidden shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-slate-400">Total Scans</span>
              <FileText className="h-5 w-5 text-sky-400" />
            </div>
            <div className="text-3xl font-bold text-slate-100">
              {historyQuery.isLoading ? "..." : totalScans}
            </div>
            <div className="text-xs text-sky-400 flex items-center gap-1 mt-2">
              <TrendingUp className="h-3 w-3" />
              All history records
            </div>
          </div>

          {/* Stat 2: Positive Detections */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 relative overflow-hidden shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-slate-400">Positive Cases</span>
              <Activity className="h-5 w-5 text-rose-400" />
            </div>
            <div className="text-3xl font-bold text-slate-100">
              {historyQuery.isLoading ? "..." : positiveScans}
            </div>
            <div className="text-xs text-rose-400 flex items-center gap-1 mt-2">
              Requires validation
            </div>
          </div>

          {/* Stat 3: Server Health */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 relative overflow-hidden shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-slate-400">API Status</span>
              <Server className="h-5 w-5 text-emerald-400" />
            </div>
            <div className="text-xl font-bold text-slate-100 capitalize">
              {healthQuery.isLoading ? "Loading..." : healthStatus}
            </div>
            <div className="flex items-center gap-2 mt-3">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  healthStatus === "healthy" ? "bg-emerald-500 animate-ping" : "bg-red-500"
                }`}
              />
              <span className="text-xs text-slate-400">FastAPI Connection</span>
            </div>
          </div>

          {/* Stat 4: Computation device */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 relative overflow-hidden shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-slate-400">Inference Device</span>
              <Activity className="h-5 w-5 text-indigo-400" />
            </div>
            <div className="text-2xl font-bold text-slate-100 uppercase">
              {healthQuery.isLoading ? "..." : deviceName}
            </div>
            <div className="text-xs text-slate-400 mt-2">
              PyTorch execution target
            </div>
          </div>
        </div>

        {/* Middle Layout (Feed and Health Alerts) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Latest Diagnosis Feed (Col-Span-2) */}
          <section className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-bold text-slate-100">Latest Diagnostic Feed</h2>
              <button
                onClick={() => router.push("/history")}
                className="text-xs text-sky-400 hover:text-sky-300 font-semibold flex items-center gap-1"
              >
                View all logs
                <ArrowUpRight className="h-4 w-4" />
              </button>
            </div>

            {historyQuery.isLoading ? (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="h-8 w-8 border-2 border-sky-400 border-t-transparent rounded-full animate-spin mb-3" />
                <p className="text-slate-400 text-sm">Querying patient records...</p>
              </div>
            ) : latestScans.length === 0 ? (
              <div className="text-center py-12 text-slate-500 border border-dashed border-slate-800 rounded-lg">
                No diagnostic scans logged yet. Navigate to Upload X-ray to start.
              </div>
            ) : (
              <div className="space-y-4">
                {latestScans.map((scan) => (
                  <div
                    key={scan.id}
                    className="flex justify-between items-center p-4 bg-slate-950 rounded-lg border border-slate-800 hover:border-slate-700 transition-colors"
                  >
                    <div>
                      <h4 className="font-semibold text-slate-200 text-sm">
                        {scan.patient_id ? `Patient ID: ${scan.patient_id}` : "Anonymous Case"}
                      </h4>
                      <p className="text-xs text-slate-400 mt-1">
                        File: {scan.image_name} | {new Date(scan.timestamp).toLocaleString()}
                      </p>
                    </div>
                    <div className="text-right">
                      {scan.detected_diseases.length > 0 ? (
                        <div className="flex flex-wrap gap-1 justify-end">
                          {scan.detected_diseases.map((d) => (
                            <span
                              key={d}
                              className="text-[10px] font-bold px-2 py-0.5 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded"
                            >
                              {d}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-[10px] font-bold px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded">
                          No Findings
                        </span>
                      )}
                      <p className="text-[10px] text-slate-400 mt-1">
                        Conf: {(scan.confidence_score * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* Clinician Instructions & Alerts */}
          <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg flex flex-col justify-between">
            <div>
              <h2 className="text-lg font-bold text-slate-100 mb-4">Clinical Guidelines</h2>
              <div className="space-y-4">
                <div className="flex gap-3 items-start text-sm text-slate-400 leading-relaxed">
                  <AlertCircle className="h-5 w-5 text-amber-500 shrink-0 mt-0.5" />
                  <p>
                    <strong>Explainability Review:</strong> Always review the Grad-CAM heatmaps on predicted classes to confirm target visual attention aligns with logical anatomy.
                  </p>
                </div>
                <div className="flex gap-3 items-start text-sm text-slate-400 leading-relaxed">
                  <AlertCircle className="h-5 w-5 text-sky-500 shrink-0 mt-0.5" />
                  <p>
                    <strong>Mock Mode:</strong> Currently running in local simulation mode. To evaluate real patients, connect to an actual database and point data paths to real PACS inputs.
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={() => router.push("/upload")}
              className="w-full bg-gradient-to-r from-sky-500 to-indigo-500 hover:from-sky-400 hover:to-indigo-400 text-white font-semibold py-3 rounded-lg text-sm text-center shadow-lg shadow-sky-500/20 transition-all duration-200 mt-6"
            >
              Analyze New Scan
            </button>
          </section>
        </div>
      </main>
    </div>
  );
}
