export type JobStatus = "failed" | "queued" | "running" | "succeeded";

export type JobType = "create_model" | "predict_model" | "train_model";

export type CreateTrainingJobRequest = {
  fitConfig?: {
    batchSize?: number;
    epochs?: number;
    shuffle?: boolean;
    validationSplit?: number;
  };
  trainingInput: {
    inputs: unknown;
    targets: unknown;
    validationInputs?: unknown;
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

export type JobProcessingOutcome =
  | {
      kind: "idle";
    }
  | {
      job: JobRecord;
      kind: "processed";
    };
