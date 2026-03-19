export type ModelStatus = "pending" | "ready" | "failed";

export type KerasModelDefinition = {
  compileConfig?: {
    loss?: Record<string, unknown> | string | string[];
    metrics?: Array<Record<string, unknown> | string>;
    optimizer?: Record<string, unknown> | string;
    runEagerly?: boolean;
  };
  format: "keras-functional" | "keras-sequential";
  modelConfig: Record<string, unknown>;
};

export type CreateModelRequest = {
  definition: KerasModelDefinition;
  modelId: string;
};

export type ModelRecord = {
  artifactPath: string;
  createdAt: string;
  definitionPath: string;
  lastPredictionAt: string | null;
  lastPredictionJobId: string | null;
  lastTrainingAt: string | null;
  lastTrainingJobId: string | null;
  modelId: string;
  predictionCount: number;
  status: ModelStatus;
  trainingCount: number;
  updatedAt: string;
};

export type CreateModelResult =
  | {
      job: import("../job/index.ts").JobRecord;
      kind: "created";
      model: ModelRecord;
    }
  | {
      kind: "conflict";
      message: string;
    };
