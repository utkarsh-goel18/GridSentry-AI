export type Health = { status: string; model_loaded: boolean; device: string; timestamp: string };
export type ModelInfo = {
  model_name: string;
  model_architecture: string;
  parameters_count: number;
  num_classes: number;
  classes: string[];
  input_spec: { num_channels: number; channels: string[]; window_length: number; sampling_rate_hz: number; nominal_freq_hz: number; duration_ms: number };
  evaluation_summary?: { test_accuracy?: number; macro_f1?: number; weighted_f1?: number } | null;
  dataset_summary?: Record<string, number> | null;
};
export type Prediction = {
  predicted_fault: string;
  fault_description: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  input_type: string;
  sample_metadata?: Record<string, unknown> | null;
  inference_time_ms: number;
};

const base = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers }
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}
export const api = {
  health: () => request<Health>("/health"),
  modelInfo: () => request<ModelInfo>("/model-info"),
  predict: (waveform: number[][], fault_index?: number) =>
    request<Prediction>("/predict", { method: "POST", body: JSON.stringify({ waveform, ...(fault_index === undefined ? {} : { fault_index }) }) })
};
export const apiBase = base;