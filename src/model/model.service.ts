/**
 * @section imports:externals
 */

import { randomUUID } from "node:crypto";
import { join } from "node:path";

/**
 * @section imports:internals
 */

import type { JobRecord } from "../job/index.ts";
import type { StorageService } from "../storage/index.ts";
import type { CreateModelRequest, CreateModelResult, DeleteModelResult, ModelRecord } from "./index.ts";

/**
 * @section class
 */

export class ModelService {
  /**
   * @section private:attributes
   */

  private readonly storageService: StorageService;

  /**
   * @section constructor
   */

  public constructor(storageService: StorageService) {
    this.storageService = storageService;
  }

  /**
   * @section factory
   */

  public static create(storageService: StorageService): ModelService {
    const service = new ModelService(storageService);
    return service;
  }

  /**
   * @section private:methods
   */

  private now(): string {
    const value = new Date().toISOString();
    return value;
  }

  private buildCreateJob(modelId: string, requestPath: string, resultPath: string, createdAt: string): JobRecord {
    const jobRecord: JobRecord = {
      createdAt,
      errorCode: null,
      errorMessage: null,
      finishedAt: null,
      jobId: randomUUID(),
      jobType: "create_model",
      modelId,
      requestPath,
      resultPath,
      startedAt: null,
      status: "queued",
    };
    return jobRecord;
  }

  /**
   * @section public:methods
   */

  public createModel(request: CreateModelRequest): CreateModelResult {
    const existingModel = this.storageService.getModelRecord(request.modelId);
    let result: CreateModelResult;

    if (existingModel) {
      result = { kind: "conflict", message: `model '${request.modelId}' already exists` };
    } else {
      const createdAt = this.now();
      const modelDirectory = this.storageService.ensureModelDirectory(request.modelId);
      const definitionPath = join(modelDirectory, "definition.json");
      const artifactPath = join(modelDirectory, "artifact.keras");
      const creationJob = this.buildCreateJob(request.modelId, "", "", createdAt);
      const jobDirectory = this.storageService.ensureJobDirectory(creationJob.jobId);
      const requestPath = join(jobDirectory, "request.json");
      const resultPath = join(jobDirectory, "result.json");
      const modelRecord: ModelRecord = {
        artifactPath,
        createdAt,
        definitionPath,
        lastTrainingAt: null,
        lastTrainingJobId: null,
        metadata: request.metadata || null,
        modelId: request.modelId,
        status: "pending",
        trainingCount: 0,
        updatedAt: createdAt,
      };
      const jobRecord: JobRecord = { ...creationJob, requestPath, resultPath };
      this.storageService.writeJsonFile(definitionPath, request.definition);
      this.storageService.writeJsonFile(requestPath, {
        artifactPath,
        definition: request.definition,
        metadata: request.metadata || null,
        modelId: request.modelId,
      });
      this.storageService.insertModelRecord({
        artifactPath,
        createdAt,
        definitionPath,
        metadata: request.metadata || null,
        modelId: request.modelId,
        status: "pending",
        updatedAt: createdAt,
      });
      this.storageService.insertJobRecord({
        createdAt,
        jobId: jobRecord.jobId,
        jobType: jobRecord.jobType,
        modelId: request.modelId,
        requestPath,
        resultPath,
        startedAt: null,
        status: "queued",
      });
      result = { job: jobRecord, kind: "created", model: modelRecord };
    }

    return result;
  }

  public listModels(): ModelRecord[] {
    const modelRecords = this.storageService.listModelRecords();
    return modelRecords;
  }

  public getModel(modelId: string): ModelRecord | null {
    const modelRecord = this.storageService.getModelRecord(modelId);
    return modelRecord;
  }

  public deleteModel(modelId: string): DeleteModelResult {
    const existingModel = this.storageService.getModelRecord(modelId);
    let result: DeleteModelResult;

    if (!existingModel) {
      result = { kind: "not_found", message: `model '${modelId}' was not found` };
    } else {
      if (this.storageService.hasActiveJobsForModel(modelId)) {
        result = { kind: "conflict", message: `model '${modelId}' has active jobs and cannot be deleted` };
      } else {
        this.storageService.deleteModel(modelId);
        result = { kind: "deleted" };
      }
    }

    return result;
  }
}
