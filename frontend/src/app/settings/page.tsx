"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Navbar from "@/components/Navbar";
import { apiService } from "@/services/api";
import {
  Server,
  Database,
  Cpu,
  Sliders,
  Shield,
  Activity,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";

export default function SettingsPage() {
  const router = useRouter();
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    // Session Guard
    if (localStorage.getItem("isLoggedIn") !== "true") {
      router.push("/login");
    }
    
    // Load local config threshold
    const saved = localStorage.getItem("predictionThreshold");
    if (saved) {
      setThreshold(parseFloat(saved));
    }
  }, [router]);

  // Query health metrics
  const healthQuery = useQuery({
    queryKey: ["systemHealthSettings"],
    queryFn: () => apiService.getHealth(),
    refetchInterval: 5000,
  });

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setThreshold(val);
    localStorage.setItem("predictionThreshold", val.toString());
  };

  const health = healthQuery.data;

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      <Navbar />

      <main className="flex-1 overflow-y-auto p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-slate-100">System Settings</h1>
          <p className="text-slate-400 mt-1">Configure diagnostic thresholds and monitor model execution states.</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Threshold Configurations */}
          <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg space-y-6">
            <div className="flex items-center gap-3">
              <Sliders className="h-6 w-6 text-sky-400" />
              <h2 className="text-lg font-bold text-slate-100">Diagnosis Parameters</h2>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed">
              Adjust the positive decision threshold. Any disease probability matching or exceeding this value will be classified as a positive detection.
            </p>

            <div className="space-y-4 pt-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-300">Positive Class Threshold</span>
                <span className="text-lg font-bold text-sky-400">{(threshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={handleThresholdChange}
                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>0.1 (High Sensitivity)</span>
                <span>0.9 (High Specificity)</span>
              </div>
            </div>
          </section>

          {/* System Health Diagnostics */}
          <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg space-y-6">
            <div className="flex items-center gap-3">
              <Shield className="h-6 w-6 text-emerald-400" />
              <h2 className="text-lg font-bold text-slate-100">Inference Health Monitor</h2>
            </div>

            {healthQuery.isLoading ? (
              <div className="flex justify-center items-center py-12">
                <div className="h-8 w-8 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
              </div>
            ) : !health ? (
              <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-sm">
                <AlertTriangle className="h-5 w-5" />
                FastAPI backend offline. Unable to query server health.
              </div>
            ) : (
              <div className="space-y-4">
                {/* 1. API connection */}
                <div className="flex justify-between items-center p-4 bg-slate-950 rounded-lg border border-slate-850">
                  <div className="flex items-center gap-3">
                    <Server className="h-5 w-5 text-slate-500" />
                    <div>
                      <h4 className="text-sm font-semibold text-slate-200">REST Backend</h4>
                      <p className="text-xs text-slate-500">http://localhost:8000/api/v1</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
                    <CheckCircle className="h-4 w-4 text-emerald-500" />
                    Connected
                  </div>
                </div>

                {/* 2. Model state */}
                <div className="flex justify-between items-center p-4 bg-slate-950 rounded-lg border border-slate-855">
                  <div className="flex items-center gap-3">
                    <Cpu className="h-5 w-5 text-slate-500" />
                    <div>
                      <h4 className="text-sm font-semibold text-slate-200">Deep Learning Model</h4>
                      <p className="text-xs text-slate-500">DenseNet121 Backbone</p>
                    </div>
                  </div>
                  <div className={`flex items-center gap-1.5 text-xs font-medium ${health.model_loaded ? "text-emerald-400" : "text-amber-500"}`}>
                    {health.model_loaded ? (
                      <>
                        <CheckCircle className="h-4 w-4 text-emerald-500" />
                        Weights Loaded
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="h-4 w-4 text-amber-500" />
                        Degraded (Mock Mode)
                      </>
                    )}
                  </div>
                </div>

                {/* 3. Computation device */}
                <div className="flex justify-between items-center p-4 bg-slate-950 rounded-lg border border-slate-860">
                  <div className="flex items-center gap-3">
                    <Activity className="h-5 w-5 text-slate-500" />
                    <div>
                      <h4 className="text-sm font-semibold text-slate-200">Device Target</h4>
                      <p className="text-xs text-slate-500">PyTorch backend target</p>
                    </div>
                  </div>
                  <span className="text-xs font-bold uppercase bg-slate-900 border border-slate-800 px-3 py-1 rounded text-slate-300">
                    {health.device}
                  </span>
                </div>

                {/* 4. SQLite state */}
                <div className="flex justify-between items-center p-4 bg-slate-950 rounded-lg border border-slate-865">
                  <div className="flex items-center gap-3">
                    <Database className="h-5 w-5 text-slate-500" />
                    <div>
                      <h4 className="text-sm font-semibold text-slate-200">History Database</h4>
                      <p className="text-xs text-slate-500">SQLite persistence layer</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
                    <CheckCircle className="h-4 w-4 text-emerald-500" />
                    Active
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
