export interface ScanApkParams {
  file: File;
  experiment: "exp1" | "exp2" | "exp3";
  threshold?: number;
}

export interface ScanResponse {
  experiment: "exp1" | "exp2" | "exp3";
  threshold: number;
  best_match: {
    family: string;
    matched_sample: string;
    score: number;
  };
  decision: {
    is_malware: boolean;
    family: string;
  };
}

export interface ExperimentPerformance {
  experiment: string;
  threshold: number;
  seen_families: string[];
  unseen_families: string[];
  overall_accuracy: number;
  seen_accuracy: number;
  unseen_accuracy: number;
}

export interface PerformanceResponse {
  exp1?: ExperimentPerformance;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

function getBackendUrl(): string {
  if (!BACKEND_URL) {
    throw new Error("NEXT_PUBLIC_BACKEND_URL is not configured in .env.local");
  }
  return BACKEND_URL;
}

export async function scanApk(params: ScanApkParams): Promise<ScanResponse> {
  if (!params.file) {
    throw new Error("File is required for scanning");
  }

  const formData = new FormData();

  // backend expects field name "apk"
  formData.append("apk", params.file);
  formData.append("experiment", params.experiment);

  if (params.threshold !== undefined) {
    formData.append("threshold", params.threshold.toString());
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(`${getBackendUrl()}/scan`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      let errorMessage = "Scan failed";

      try {
        const errorData = await response.json();
        errorMessage =
          errorData?.detail ||
          errorData?.message ||
          errorMessage;
      } catch {
        errorMessage = await response.text();
      }

      throw new Error(errorMessage);
    }

    return (await response.json()) as ScanResponse;
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error("Request timed out after 30 seconds");
    }

    if (err instanceof SyntaxError) {
      throw new Error("Invalid response format from server");
    }

    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function getPerformance(): Promise<PerformanceResponse> {
  const response = await fetch(`${getBackendUrl()}/performance`);

  if (!response.ok) {
    let errorMessage = "Failed to fetch performance";

    try {
      const errorData = await response.json();
      errorMessage =
        errorData?.detail ||
        errorData?.message ||
        errorMessage;
    } catch {
      errorMessage = await response.text();
    }

    throw new Error(errorMessage);
  }

  return (await response.json()) as PerformanceResponse;
}