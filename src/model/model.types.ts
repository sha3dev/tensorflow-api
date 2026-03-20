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
  metadata?: Record<string, unknown>;
  modelId: string;
};

export type ModelRecord = {
  artifactPath: string;
  createdAt: string;
  definitionPath: string;
  lastTrainingAt: string | null;
  lastTrainingJobId: string | null;
  metadata: Record<string, unknown> | null;
  modelId: string;
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

export type DeleteModelResult =
  | {
      kind: "deleted";
    }
  | {
      kind: "conflict" | "not_found";
      message: string;
    };

export type UpdateModelMetadataResult =
  | {
      kind: "not_found";
      message: string;
    }
  | {
      kind: "updated";
      model: ModelRecord;
    };
