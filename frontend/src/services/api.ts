import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

export interface PredictionResult {
  id?: number;
  patient_id?: string;
  image_name: string;
  predictions: Record<string, number>;
  detected_diseases: string[];
  confidence_score: number;
  timestamp: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
  database_connected: boolean;
}

export const apiService = {
  /**
   * Fetches backend database and model load status.
   */
  async getHealth(): Promise<HealthStatus> {
    const response = await apiClient.get<HealthStatus>("/health");
    return response.data;
  },

  /**
   * Fetches prediction history records, optionally filtered by patient ID.
   */
  async getHistory(patientId?: string): Promise<PredictionResult[]> {
    const params = patientId ? { patient_id: patientId } : {};
    const response = await apiClient.get<PredictionResult[]>("/history", { params });
    return response.data;
  },

  /**
   * Uploads X-ray image for multi-label classification.
   */
  async predict(file: File, patientId?: string): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append("file", file);
    if (patientId) {
      formData.append("patient_id", patientId);
    }

    const response = await apiClient.post<PredictionResult>("/predict", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },

  /**
   * Uploads X-ray image and class name to generate a Grad-CAM overlay.
   * Returns a local object URL pointing to the returned image Blob.
   */
  async getGradcamOverlay(file: File, targetClass: string): Promise<string> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_class", targetClass);

    const response = await apiClient.post("/gradcam", formData, {
      responseType: "blob",
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    // Create a local blob URL
    const blob = new Blob([response.data], { type: "image/png" });
    return URL.createObjectURL(blob);
  },
};
