"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function RootPage() {
  const router = useRouter();

  useEffect(() => {
    const isLoggedIn = localStorage.getItem("isLoggedIn");
    if (isLoggedIn === "true") {
      router.push("/dashboard");
    } else {
      router.push("/login");
    }
  }, [router]);

  return (
    <main className="min-h-screen bg-slate-950 flex items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <div className="h-10 w-10 border-4 border-sky-400 border-t-transparent rounded-full animate-spin" />
        <p className="text-slate-400 text-sm font-medium">Redirecting to session portal...</p>
      </div>
    </main>
  );
}
