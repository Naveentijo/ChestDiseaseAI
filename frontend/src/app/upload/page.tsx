"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import Navbar from "@/components/Navbar";
import { apiService, PredictionResult } from "@/services/api";
import {
  Upload,
  File,
  X,
  AlertCircle,
  Activity,
  ChevronRight,
  Eye,
  CheckCircle,
} from "lucide-react";

export default function UploadPage() {
  const router = useRouter();
  const queryClient = useQueryClient();

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [patientId, setPatientId] = useState("");
  const [error, setError] = useState("");
  
  // GradCAM overlay states
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [selectedClass, setSelectedClass] = useState<string | null>(null);
  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);

  useEffect(() => {
    // Session Guard
    if (localStorage.getItem("isLoggedIn") !== "true") {
      router.push("/login");
    }
  }, [router]);

  // Clean up Object URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (gradcamUrl) URL.revokeObjectURL(gradcamUrl);
    };
  }, [previewUrl, gradcamUrl]);

  // Prediction Mutation
  const predictMutation = useMutation({
    mutationFn: (data: { file: File; patientId?: string }) =>
      apiService.predict(data.file, data.patientId),
    onSuccess: (data) => {
      setPrediction(data);
      // Invalidate history cache so dashboard and logs tables update
      queryClient.invalidateQueries({ queryKey: ["predictionHistory"] });
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || "An error occurred during model inference.");
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError("");
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (!selectedFile.type.startsWith("image/")) {
        setError("Invalid file format. Please upload a chest X-ray image.");
        return;
      }
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      // Reset prediction states
      setPrediction(null);
      setSelectedClass(null);
      setGradcamUrl(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setError("");
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      if (!droppedFile.type.startsWith("image/")) {
        setError("Invalid file format. Please upload a chest X-ray image.");
        return;
      }
      setFile(droppedFile);
      setPreviewUrl(URL.createObjectURL(droppedFile));
      setPrediction(null);
      setSelectedClass(null);
      setGradcamUrl(null);
    }
  };

  const removeFile = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setPrediction(null);
    setSelectedClass(null);
    setGradcamUrl(null);
  };

  const handleUploadSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image file to analyze.");
      return;
    }
    predictMutation.mutate({ file, patientId: patientId || undefined });
  };

  // Fetch Grad-CAM Overlay when a class is selected
  const handleClassSelect = async (className: string) => {
    if (!file) return;
    setSelectedClass(className);
    setGradcamLoading(true);
    setError("");
    
    try {
      if (gradcamUrl) {
        URL.revokeObjectURL(gradcamUrl);
      }
      const url = await apiService.getGradcamOverlay(file, className);
      setGradcamUrl(url);
    } catch (err: any) {
      setError("Failed to generate Grad-CAM explainability heatmap.");
    } finally {
      setGradcamLoading(false);
    }
  };

  const handleReset = () => {
    removeFile();
    setPatientId("");
  };

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      <Navbar />

      <main className="flex-1 overflow-y-auto p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-slate-100">Diagnostic Suite</h1>
          <p className="text-slate-400 mt-1">
            Analyze Chest X-rays using deep learning models and visualize feature focus.
          </p>
        </header>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm p-4 rounded-xl mb-6 flex gap-3 items-center">
            <AlertCircle className="h-5 w-5 shrink-0" />
            {error}
          </div>
        )}

        {/* Dynamic Workflow Rendering */}
        {!prediction ? (
          /* View 1: Upload Dropzone Form */
          <div className="max-w-2xl bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-lg">
            <form onSubmit={handleUploadSubmit} className="space-y-6">
              {/* Patient ID Input */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Patient Reference ID (Optional)
                </label>
                <input
                  type="text"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="EX: PATIENT-4091"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg py-3 px-4 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-sky-500"
                />
              </div>

              {/* Drag & Drop Zone */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Chest X-ray Image (DICOM / JPEG / PNG)
                </label>
                {!previewUrl ? (
                  <div
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    className="border-2 border-dashed border-slate-800 hover:border-sky-500/50 bg-slate-950/50 rounded-xl p-12 text-center cursor-pointer transition-colors duration-200"
                    onClick={() => document.getElementById("file-input")?.click()}
                  >
                    <Upload className="h-10 w-10 text-slate-500 mx-auto mb-4" />
                    <p className="text-sm text-slate-300 font-medium">
                      Drag & Drop chest X-ray or{" "}
                      <span className="text-sky-400">browse local files</span>
                    </p>
                    <p className="text-xs text-slate-500 mt-2">
                      Supports PNG, JPG, JPEG up to 10MB
                    </p>
                    <input
                      id="file-input"
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </div>
                ) : (
                  <div className="relative bg-slate-950 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <File className="h-8 w-8 text-sky-400" />
                      <div className="max-w-md truncate">
                        <p className="text-sm font-medium text-slate-200 truncate">
                          {file?.name}
                        </p>
                        <p className="text-xs text-slate-500">
                          {file ? (file.size / 1024 / 1024).toFixed(2) : 0} MB
                        </p>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={removeFile}
                      className="p-1 text-slate-400 hover:bg-slate-800 rounded-full"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                )}
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={predictMutation.isPending || !file}
                className="w-full bg-gradient-to-r from-sky-500 to-indigo-500 hover:from-sky-400 hover:to-indigo-400 text-white font-semibold py-3 rounded-lg text-sm flex items-center justify-center gap-2 shadow-lg shadow-sky-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {predictMutation.isPending ? (
                  <>
                    <Activity className="h-5 w-5 animate-spin" />
                    Deep Learning Inference Running...
                  </>
                ) : (
                  <>
                    Upload and Analyze Scan
                    <ChevronRight className="h-5 w-5" />
                  </>
                )}
              </button>
            </form>
          </div>
        ) : (
          /* View 2: Prediction results & Grad-CAM viewer */
          <div className="space-y-8">
            {/* Action Bar */}
            <div className="flex justify-between items-center">
              <span className="text-slate-400 text-sm">
                Analysis complete for: <strong>{prediction.image_name}</strong>
              </span>
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-slate-900 border border-slate-800 hover:bg-slate-800 text-sm font-medium rounded-lg text-slate-300 transition-colors"
              >
                Analyze Another Scan
              </button>
            </div>

            {/* Split Grid layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Left Column: Diagnostics list */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg space-y-6">
                <div>
                  <h2 className="text-lg font-bold text-slate-100">Classification Probabilities</h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Disease predictions computed by DenseNet121. Select a class to load Grad-CAM explainability maps.
                  </p>
                </div>

                <div className="space-y-4">
                  {Object.entries(prediction.predictions).map(([name, val]) => {
                    const isDetected = val >= 0.5;
                    const isSelected = selectedClass === name;
                    
                    return (
                      <div
                        key={name}
                        onClick={() => handleClassSelect(name)}
                        className={`p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
                          isSelected
                            ? "bg-sky-500/10 border-sky-500"
                            : "bg-slate-950 border-slate-800 hover:border-slate-700"
                        }`}
                      >
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-semibold text-sm text-slate-200">{name}</span>
                          <div className="flex items-center gap-2">
                            {isDetected && (
                              <span className="text-[10px] font-bold px-2 py-0.5 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded">
                                Detected
                              </span>
                            )}
                            <span className="text-sm font-bold text-slate-100">
                              {(val * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>

                        {/* Custom visual progress bar */}
                        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${isDetected ? "bg-rose-500" : "bg-sky-500"}`}
                            style={{ width: `${val * 100}%` }}
                          />
                        </div>
                        
                        <div className="flex justify-end mt-2 text-[10px] text-slate-500 items-center gap-1">
                          <Eye className="h-3.5 w-3.5" />
                          Click to load feature activation map
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Right Column: Dynamic side-by-side image viewer */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg flex flex-col items-center justify-center">
                <h3 className="text-sm font-bold text-slate-200 mb-4 w-full text-left">
                  {selectedClass ? `Grad-CAM explainability overlay: ${selectedClass}` : "Patient Chest X-ray Preview"}
                </h3>

                <div className="relative border border-slate-800 rounded-lg overflow-hidden bg-slate-950 w-full aspect-square flex items-center justify-center">
                  {gradcamLoading ? (
                    <div className="absolute inset-0 bg-slate-950/80 flex flex-col items-center justify-center z-10">
                      <div className="h-10 w-10 border-4 border-sky-400 border-t-transparent rounded-full animate-spin mb-3" />
                      <p className="text-sm text-slate-400 font-medium">Generating heatmap overlay...</p>
                    </div>
                  ) : null}

                  {selectedClass && gradcamUrl ? (
                    <img
                      src={gradcamUrl}
                      alt="GradCAM Overlay"
                      className="w-full h-full object-contain"
                    />
                  ) : previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Original Scan"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-slate-600 text-sm">No image available</div>
                  )}
                </div>

                {!selectedClass && (
                  <p className="text-xs text-slate-500 mt-4 text-center">
                    Select a disease class from the list on the left to compute and overlay a feature activation heatmap showing where the model focused.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
