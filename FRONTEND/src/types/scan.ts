export type ScanResponse = {
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

export type ExperimentPerformance = {
  experiment: string;
  threshold: number;
  seen_families: string[];
  unseen_families: string[];
  overall_accuracy: number;
  seen_accuracy: number;
  unseen_accuracy: number;
};

export type PerformanceResponse = {
  exp1?: ExperimentPerformance;
};