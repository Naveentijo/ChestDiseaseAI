"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Activity, Lock, Mail, ArrowRight } from "lucide-react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    // If already logged in, redirect to dashboard
    if (localStorage.getItem("isLoggedIn") === "true") {
      router.push("/dashboard");
    }
  }, [router]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    // Simulated login delay
    setTimeout(() => {
      if (email && password) {
        localStorage.setItem("isLoggedIn", "true");
        router.push("/dashboard");
      } else {
        setError("Please enter valid clinician credentials.");
        setLoading(false);
      }
    }, 800);
  };

  return (
    <main className="min-h-screen bg-slate-950 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background ambient glow */}
      <div className="absolute top-1/4 left-1/4 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 translate-x-1/2 translate-y-1/2 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />

      <div className="w-full max-w-md bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-2xl p-8 shadow-xl relative z-10">
        {/* Brand Header */}
        <div className="flex flex-col items-center mb-8">
          <div className="h-12 w-12 bg-sky-500/10 rounded-xl flex items-center justify-center mb-4 border border-sky-500/20">
            <Activity className="h-6 w-6 text-sky-400 animate-pulse" />
          </div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
            Clinician Portal
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Access ChestDiseaseAI Diagnostic Suite
          </p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm p-3 rounded-lg mb-6 text-center">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Email input */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Work Email
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="radiologist@clinic.org"
                className="w-full bg-slate-950 border border-slate-800 rounded-lg py-3 pl-11 pr-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 transition-colors"
              />
            </div>
          </div>

          {/* Password input */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Secure Key
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-slate-950 border border-slate-800 rounded-lg py-3 pl-11 pr-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 transition-colors"
              />
            </div>
          </div>

          {/* Submit button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-sky-500 to-indigo-500 hover:from-sky-400 hover:to-indigo-400 text-white font-semibold py-3 rounded-lg flex items-center justify-center gap-2 transition-all duration-200 shadow-lg shadow-sky-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="h-5 w-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <>
                Initialize Diagnoses
                <ArrowRight className="h-5 w-5" />
              </>
            )}
          </button>
        </form>

        {/* Register link */}
        <div className="mt-8 text-center text-sm text-slate-400">
          New clinician?{" "}
          <Link href="/register" className="text-sky-400 hover:text-sky-300 font-medium">
            Register for access
          </Link>
        </div>
      </div>
    </main>
  );
}
