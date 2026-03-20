export type JobStatus = "failed" | "queued" | "running" | "succeeded";

export type JobType = "create_model" | "train_model";

export type TrainingSampleWeight = number[] | Record<string, number[]>;

export type ShapeSummaryRecord = {
  [key: string]: ShapeSummary;
};

export type ShapeSummary = ShapeSummaryRecord | number[];

export type TrainingInputSummary = {
  sampleWeightKeys: string[];
  sampleWeightShapes: ShapeSummary | null;
  targetKeys: string[];
  targetShapes: ShapeSummary | null;
  validationSampleWeightKeys: string[];
  validationSampleWeightShapes: ShapeSummary | null;
  validationTargetKeys: string[];
  validationTargetShapes: ShapeSummary | null;
};

export type JobFailureDiagnostics = {
  modelOutputCount: number;
  modelOutputNames: string[];
  pythonExceptionType: string;
  stderrTail: string;
  traceback: string;
  trainingInputSummary: TrainingInputSummary;
};

export type CreateTrainingJobRequest = {
  fitConfig?: {
    batchSize?: number;
    epochs?: number;
    shuffle?: boolean;
    validationSplit?: number;
  };
  modelMetadata?: Record<string, unknown>;
  trainingInput: {
    inputs: unknown;
    sampleWeights?: TrainingSampleWeight;
    targets: unknown;
    validationInputs?: unknown;
    validationSampleWeights?: TrainingSampleWeight;
    validationTargets?: unknown;
  };
};

export type CreatePredictionJobRequest = {
  predictionInput: {
    inputs: unknown;
  };
};

export type JobRecord = {
  createdAt: string;
  diagnostics?: JobFailureDiagnostics | null;
  errorCode: string | null;
  errorMessage: string | null;
  finishedAt: string | null;
  jobId: string;
  jobType: JobType;
  modelId: string;
  requestPath: string;
  resultPath: string;
  startedAt: string | null;
  status: JobStatus;
};

export type JobListFilter = {
  limit?: number;
  modelId?: string;
  status?: JobStatus;
};

export type JobResultPayload = Record<string, unknown>;

export type FailedJobResultPayload = {
  diagnostics?: JobFailureDiagnostics;
  errorCode: string;
  errorMessage: string;
  modelId: string;
  status: "failed";
};

export type CreateJobResult =
  | {
      job: JobRecord;
      kind: "created";
    }
  | {
      kind: "conflict" | "not_found";
      message: string;
    };

export type JobResultLookup =
  | {
      job: JobRecord;
      kind: "failed";
      result: FailedJobResultPayload;
    }
  | {
      job: JobRecord;
      kind: "not_ready";
    }
  | {
      kind: "not_found";
      message: string;
    }
  | {
      job: JobRecord;
      kind: "ready";
      result: JobResultPayload;
    };

export type PredictionExecutionResult =
  | {
      kind: "completed";
      result: PredictionResultPayload;
    }
  | {
      kind: "conflict" | "failed" | "not_found";
      message: string;
    };

export type PredictionResultPayload = {
  modelId: string;
  outputs: Record<string, unknown> | unknown;
  status: "predicted";
};

export type TrainingJobResultPayload = {
  history?: Record<string, unknown>;
  modelId: string;
  status: "succeeded";
  trainedAt: string;
};

export type JobProcessingOutcome =
  | {
      kind: "idle";
    }
  | {
      job: JobRecord;
      kind: "processed";
    };
