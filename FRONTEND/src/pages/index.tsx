import { useEffect, useMemo, useState } from "react";
import { getPerformance, scanApk } from "@/lib/api";

type ScanResponse = {
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
};

type ExperimentPerformance = {
  experiment: string;
  threshold: number;
  seen_families: string[];
  unseen_families: string[];
  overall_accuracy: number;
  seen_accuracy: number;
  unseen_accuracy: number;
};

type PerformanceResponse = {
  exp1?: ExperimentPerformance;
};

function safeErrorMessage(err: unknown): string {
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;
  if (err && typeof err === "object") {
    try {
      return JSON.stringify(err);
    } catch {
      return "An unexpected error occurred";
    }
  }
  return "An unexpected error occurred";
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [experiment, setExperiment] = useState<"exp1" | "exp2" | "exp3">("exp1");
  const [useCustom, setUseCustom] = useState(false);
  const [threshold, setThreshold] = useState<string>("");

  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ScanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [performance, setPerformance] = useState<PerformanceResponse | null>(null);
  const [perfLoading, setPerfLoading] = useState(true);
  const [perfError, setPerfError] = useState<string | null>(null);

  const recommended = useMemo(() => 0.3, []);

  useEffect(() => {
    async function loadPerformance() {
      try {
        setPerfLoading(true);
        setPerfError(null);
        const res = await getPerformance();
        setPerformance(res);
      } catch (e) {
        setPerfError(safeErrorMessage(e));
      } finally {
        setPerfLoading(false);
      }
    }

    loadPerformance();
  }, []);

  async function onScan() {
    setError(null);
    setData(null);

    if (!file) {
      setError("Please select an APK file first.");
      return;
    }

    setLoading(true);

    try {
      const th = useCustom && threshold.trim() ? Number(threshold) : undefined;
      const res = await scanApk({
        file,
        experiment,
        threshold: th,
      });
      setData(res as ScanResponse);
    } catch (e) {
      setError(safeErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  const perf = performance?.exp1;

  return (
    <div className="container">
      <div style={{ marginBottom: 22 }}>
        <div
          style={{
            fontSize: 34,
            fontWeight: 900,
            letterSpacing: "-0.03em",
            lineHeight: 1.1,
            marginBottom: 8,
          }}
        >
          Android Malware Detection
        </div>
        <div style={{ fontSize: 16, color: "var(--muted)", marginBottom: 6 }}>
          Thesis Dashboard
        </div>
        <div className="small">
          Upload APK → grayscale conversion → support-set comparison → threshold-based decision
        </div>
      </div>

      <div className="grid">
        <div style={{ display: "grid", gap: 16 }}>
          <div className="card">
            <div className="sectionTitle">Upload APK</div>

            <input
              className="input"
              type="file"
              accept=".apk"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />

            <div className="fileBox" style={{ marginTop: 12 }}>
              <div className="small" style={{ marginBottom: 4 }}>
                Selected File
              </div>
              <div className="fileName">{file ? file.name : "No file selected"}</div>
            </div>
          </div>

          <div className="card">
            <div className="sectionTitle">Experiment</div>

            <select
              className="select"
              value={experiment}
              onChange={(e) =>
                setExperiment(e.target.value as "exp1" | "exp2" | "exp3")
              }
            >
              <option value="exp1">Exp1 — CNN4</option>
              <option value="exp2">Exp2 — CNN6 + Dropout</option>
              <option value="exp3">Exp3 — ResNet34</option>
            </select>

            <div className="small" style={{ marginTop: 10 }}>
              Right now only <b>Exp1</b> is ready in backend, so keep Exp1 selected.
            </div>
          </div>

          <div className="card">
            <div className="sectionTitle">Threshold</div>

            <label className="small" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input
                type="checkbox"
                checked={useCustom}
                onChange={() => setUseCustom(!useCustom)}
              />
              Use custom threshold
            </label>

            <input
              className="input"
              type="number"
              step="0.01"
              disabled={!useCustom}
              placeholder={`Recommended: ${recommended}`}
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              style={{ marginTop: 12 }}
            />
          </div>

          <div className="card">
            <button className="button primaryButton" onClick={onScan} disabled={loading}>
              {loading ? "Scanning APK..." : "Scan APK"}
            </button>

            <div className="small" style={{ marginTop: 10 }}>
              Backend: <b>{process.env.NEXT_PUBLIC_BACKEND_URL || "Not configured"}</b>
            </div>
          </div>
        </div>

        <div style={{ display: "grid", gap: 16 }}>
          <div className="card heroCard">
            <div className="sectionTitle">Scan Result</div>

            {!loading && !error && !data && (
              <div className="emptyState">
                <div className="emptyIcon">🛡️</div>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>No scan result yet</div>
                <div className="small">Upload an APK and click Scan APK to see the prediction.</div>
              </div>
            )}

            {loading && (
              <div className="emptyState">
                <div className="emptyIcon">⏳</div>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Scanning in progress</div>
                <div className="small">Please wait while the model compares the APK with support embeddings.</div>
              </div>
            )}

            {error && (
              <div className="resultAlert danger">
                <div className="resultAlertTitle">Scan Failed</div>
                <div className="resultAlertText">{error}</div>
              </div>
            )}

            {data && (
              <>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 12,
                    flexWrap: "wrap",
                    marginBottom: 16,
                  }}
                >
                  <div>
                    {data.decision.is_malware ? (
                      <span className="resultBadge danger">Malware Detected</span>
                    ) : (
                      <span className="resultBadge safe">Benign / Unknown</span>
                    )}
                  </div>

                  <div className="miniPill">
                    Experiment: <b>{data.experiment.toUpperCase()}</b>
                  </div>
                </div>

                <div className="bigResultCard">
                  <div className="bigResultLabel">Predicted Family</div>
                  <div className="bigResultValue">{data.decision.family}</div>
                </div>

                <div className="statsGrid" style={{ marginTop: 14 }}>
                  <div className="statTile">
                    <div className="statLabel">Best Match</div>
                    <div className="statValueSmall">
                      {data.best_match.family} / {data.best_match.matched_sample}
                    </div>
                  </div>

                  <div className="statTile">
                    <div className="statLabel">Distance Score</div>
                    <div className="statValueSmall">{data.best_match.score}</div>
                  </div>

                  <div className="statTile">
                    <div className="statLabel">Threshold</div>
                    <div className="statValueSmall">{data.threshold}</div>
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="card">
            <div className="sectionTitle">Experiment Performance (EXP1)</div>

            {perfLoading && <div className="small">Loading performance...</div>}

            {perfError && (
              <div className="resultAlert danger" style={{ marginTop: 12 }}>
                <div className="resultAlertTitle">Performance Error</div>
                <div className="resultAlertText">{perfError}</div>
              </div>
            )}

            {!perfLoading && !perfError && perf && (
              <>
                <div className="statsGrid">
                  <div className="statTile">
                    <div className="statLabel">Model</div>
                    <div className="statValue">CNN4</div>
                  </div>

                  <div className="statTile">
                    <div className="statLabel">Overall Accuracy</div>
                    <div className="statValue">{perf.overall_accuracy}%</div>
                  </div>

                  <div className="statTile">
                    <div className="statLabel">Seen Accuracy</div>
                    <div className="statValue">{perf.seen_accuracy}%</div>
                  </div>

                  <div className="statTile">
                    <div className="statLabel">Unseen Accuracy</div>
                    <div className="statValue">{perf.unseen_accuracy}%</div>
                  </div>
                </div>

                <div className="familyInfoWrap">
                  <div className="familyInfoCard">
                    <div className="familyInfoTitle">Seen Families</div>
                    <div className="familyInfoText">{perf.seen_families.join(", ")}</div>
                  </div>

                  <div className="familyInfoCard">
                    <div className="familyInfoTitle">Unseen Families</div>
                    <div className="familyInfoText">{perf.unseen_families.join(", ")}</div>
                  </div>
                </div>
              </>
            )}

            {!perfLoading && !perfError && !perf && (
              <div className="small">No Exp1 performance data found.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}