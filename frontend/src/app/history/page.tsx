"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Navbar from "@/components/Navbar";
import { apiService } from "@/services/api";
import { Search, Calendar, User, FileText, ArrowRight } from "lucide-react";

export default function HistoryPage() {
  const router = useRouter();
  const [searchId, setSearchId] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");

  useEffect(() => {
    // Session Guard
    if (localStorage.getItem("isLoggedIn") !== "true") {
      router.push("/login");
    }
  }, [router]);

  // Debounce search input to avoid spamming network requests
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearch(searchId);
    }, 400);
    return () => clearTimeout(handler);
  }, [searchId]);

  // Query prediction history based on search filters
  const historyQuery = useQuery({
    queryKey: ["predictionHistory", debouncedSearch],
    queryFn: () => apiService.getHistory(debouncedSearch || undefined),
  });

  const records = historyQuery.data || [];

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      <Navbar />

      <main className="flex-1 overflow-y-auto p-8">
        <header className="mb-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-100">Diagnosis History</h1>
            <p className="text-slate-400 mt-1">Audit log of all chest X-ray scans and model outputs.</p>
          </div>
          
          {/* Search bar */}
          <div className="relative w-full md:w-80">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input
              type="text"
              value={searchId}
              onChange={(e) => setSearchId(e.target.value)}
              placeholder="Search by Patient ID..."
              className="w-full bg-slate-900 border border-slate-800 rounded-lg py-2 pl-10 pr-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500"
            />
          </div>
        </header>

        {/* History Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
          {historyQuery.isLoading ? (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="h-10 w-10 border-4 border-sky-400 border-t-transparent rounded-full animate-spin mb-4" />
              <p className="text-slate-400 text-sm font-medium">Fetching history logs...</p>
            </div>
          ) : records.length === 0 ? (
            <div className="text-center py-24 text-slate-500">
              No historical records found matching filter.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                    <th className="p-4 text-left">Timestamp</th>
                    <th className="p-4 text-left">Patient Reference</th>
                    <th className="p-4 text-left">X-ray Filename</th>
                    <th className="p-4 text-left">Detected Conditions</th>
                    <th className="p-4 text-right">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50 text-sm text-slate-300">
                  {records.map((row) => (
                    <tr key={row.id} className="hover:bg-slate-800/30 transition-colors">
                      {/* Timestamp */}
                      <td className="p-4 whitespace-nowrap text-slate-400">
                        <div className="flex items-center gap-2">
                          <Calendar className="h-4 w-4 text-slate-500" />
                          {new Date(row.timestamp).toLocaleString()}
                        </div>
                      </td>
                      
                      {/* Patient ID */}
                      <td className="p-4 whitespace-nowrap font-medium text-slate-200">
                        <div className="flex items-center gap-2">
                          <User className="h-4 w-4 text-slate-500" />
                          {row.patient_id ? row.patient_id : <span className="text-slate-500 italic">Anonymous</span>}
                        </div>
                      </td>
                      
                      {/* File Name */}
                      <td className="p-4 whitespace-nowrap truncate max-w-xs text-slate-400">
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-slate-500" />
                          {row.image_name}
                        </div>
                      </td>
                      
                      {/* Conditions Badges */}
                      <td className="p-4">
                        {row.detected_diseases.length > 0 ? (
                          <div className="flex flex-wrap gap-1.5">
                            {row.detected_diseases.map((d) => (
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
                      </td>
                      
                      {/* Confidence */}
                      <td className="p-4 text-right whitespace-nowrap font-bold text-slate-200">
                        {(row.confidence_score * 100).toFixed(0)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
