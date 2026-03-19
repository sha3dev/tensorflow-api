/**
 * @section imports:externals
 */

import { randomUUID } from "node:crypto";
import { join } from "node:path";

/**
 * @section imports:internals
 */

import type { ModelService } from "../model/index.ts";
import type { PythonRuntimeService } from "../python-runtime/index.ts";
import type { StorageService } from "../storage/index.ts";
import type {
  CreateJobResult,
  CreatePredictionJobRequest,
  CreateTrainingJobRequest,
  JobListFilter,
  JobProcessingOutcome,
  JobRecord,
  JobResultLookup,
} from "./index.ts";

/**
 * @section class
 */

export class JobService {
  /**
   * @section private:attributes
   */

  private readonly modelService: ModelService;

  private readonly pythonRuntimeService: PythonRuntimeService;

  private readonly storageService: StorageService;

  /**
   * @section constructor
   */

  public constructor(storageService: StorageService, modelService: ModelService, pythonRuntimeService: PythonRuntimeService) {
    this.storageService = storageService;
    this.modelService = modelService;
    this.pythonRuntimeService = pythonRuntimeService;
  }

  /**
   * @section factory
   */

  public static create(storageService: StorageService, modelService: ModelService, pythonRuntimeService: PythonRuntimeService): JobService {
    const service = new JobService(storageService, modelService, pythonRuntimeService);
    return service;
  }

  /**
   * @section private:methods
   */

  private now(): string {
    const value = new Date().toISOString();
    return value;
  }

  private createQueuedJobRecord(jobType: JobRecord["jobType"], modelId: string, requestPath: string, resultPath: string, createdAt: string): JobRecord {
    const jobRecord: JobRecord = {
      createdAt,
      errorCode: null,
      errorMessage: null,
      finishedAt: null,
      jobId: randomUUID(),
      jobType,
      modelId,
      requestPath,
      resultPath,
      startedAt: null,
      status: "queued",
    };
    return jobRecord;
  }

  private mapJobTypeToAction(jobType: JobRecord["jobType"]): "create-model" | "predict-model" | "train-model" {
    let action: "create-model" | "predict-model" | "train-model";

    if (jobType === "create_model") {
      action = "create-model";
    } else {
      if (jobType === "train_model") {
        action = "train-model";
      } else {
        action = "predict-model";
      }
    }

    return action;
  }

  private updateModelAfterSuccess(jobRecord: JobRecord, finishedAt: string): void {
    if (jobRecord.jobType === "create_model") {
      this.storageService.markModelReady(jobRecord.modelId, finishedAt);
    }

    if (jobRecord.jobType === "train_model") {
      this.storageService.markModelTrainingSucceeded(jobRecord.modelId, jobRecord.jobId, finishedAt);
    }

    if (jobRecord.jobType === "predict_model") {
      this.storageService.markModelPredictionSucceeded(jobRecord.modelId, jobRecord.jobId, finishedAt);
    }
  }

  /**
   * @section public:methods
   */

  public recoverInterruptedJobs(): void {
    this.storageService.markRunningJobsFailedAfterRestart(this.now());
  }

  public enqueueTrainingJob(modelId: string, request: CreateTrainingJobRequest): CreateJobResult {
    const modelRecord = this.modelService.getModel(modelId);
    let result: CreateJobResult;

    if (!modelRecord) {
      result = { kind: "not_found", message: `model '${modelId}' was not found` };
    } else {
      if (modelRecord.status !== "ready") {
        result = { kind: "conflict", message: `model '${modelId}' is not ready for training` };
      } else {
        const createdAt = this.now();
        const jobRecord = this.createQueuedJobRecord("train_model", modelId, "", "", createdAt);
        const jobDirectory = this.storageService.ensureJobDirectory(jobRecord.jobId);
        const requestPath = join(jobDirectory, "request.json");
        const resultPath = join(jobDirectory, "result.json");
        const queuedJob: JobRecord = { ...jobRecord, requestPath, resultPath };
        this.storageService.writeJsonFile(requestPath, {
          artifactPath: modelRecord.artifactPath,
          fitConfig: request.fitConfig || {},
          modelId,
          trainingInput: request.trainingInput,
        });
        this.storageService.insertJobRecord({
          createdAt,
          jobId: queuedJob.jobId,
          jobType: queuedJob.jobType,
          modelId,
          requestPath,
          resultPath,
          status: "queued",
        });
        this.storageService.markModelTrainingQueued(modelId, queuedJob.jobId, createdAt);
        result = { job: queuedJob, kind: "created" };
      }
    }

    return result;
  }

  public enqueuePredictionJob(modelId: string, request: CreatePredictionJobRequest): CreateJobResult {
    const modelRecord = this.modelService.getModel(modelId);
    let result: CreateJobResult;

    if (!modelRecord) {
      result = { kind: "not_found", message: `model '${modelId}' was not found` };
    } else {
      if (modelRecord.status !== "ready") {
        result = { kind: "conflict", message: `model '${modelId}' is not ready for prediction` };
      } else {
        const createdAt = this.now();
        const jobRecord = this.createQueuedJobRecord("predict_model", modelId, "", "", createdAt);
        const jobDirectory = this.storageService.ensureJobDirectory(jobRecord.jobId);
        const requestPath = join(jobDirectory, "request.json");
        const resultPath = join(jobDirectory, "result.json");
        const queuedJob: JobRecord = { ...jobRecord, requestPath, resultPath };
        this.storageService.writeJsonFile(requestPath, {
          artifactPath: modelRecord.artifactPath,
          modelId,
          predictionInput: request.predictionInput,
        });
        this.storageService.insertJobRecord({
          createdAt,
          jobId: queuedJob.jobId,
          jobType: queuedJob.jobType,
          modelId,
          requestPath,
          resultPath,
          status: "queued",
        });
        this.storageService.markModelPredictionQueued(modelId, queuedJob.jobId, createdAt);
        result = { job: queuedJob, kind: "created" };
      }
    }

    return result;
  }

  public listJobs(filter?: JobListFilter): JobRecord[] {
    const jobRecords = this.storageService.listJobRecords(filter);
    return jobRecords;
  }

  public getJob(jobId: string): JobRecord | null {
    const jobRecord = this.storageService.getJobRecord(jobId);
    return jobRecord;
  }

  public getJobResult(jobId: string): JobResultLookup {
    const jobRecord = this.getJob(jobId);
    let result: JobResultLookup;

    if (!jobRecord) {
      result = { kind: "not_found", message: `job '${jobId}' was not found` };
    } else {
      if (jobRecord.status !== "succeeded") {
        result = { job: jobRecord, kind: "not_ready" };
      } else {
        const jobResult = this.storageService.readJobResult(jobRecord.resultPath) || {};
        result = { job: jobRecord, kind: "ready", result: jobResult };
      }
    }

    return result;
  }

  public async processNextQueuedJob(): Promise<JobProcessingOutcome> {
    const claimedJob = this.storageService.claimNextQueuedJob(this.now());
    let outcome: JobProcessingOutcome;

    if (!claimedJob) {
      outcome = { kind: "idle" };
    } else {
      const executionResult = await this.pythonRuntimeService.execute({
        action: this.mapJobTypeToAction(claimedJob.jobType),
        requestPath: claimedJob.requestPath,
        resultPath: claimedJob.resultPath,
      });
      const finishedAt = this.now();

      if (executionResult.isSuccess) {
        this.storageService.markJobSucceeded(claimedJob.jobId, finishedAt);
        this.updateModelAfterSuccess(claimedJob, finishedAt);
      } else {
        this.storageService.markJobFailed(claimedJob.jobId, finishedAt, "internal_error", executionResult.errorMessage || "python worker execution failed");

        if (claimedJob.jobType === "create_model") {
          this.storageService.markModelFailed(claimedJob.modelId, finishedAt);
        }
      }

      outcome = { job: this.getJob(claimedJob.jobId) || claimedJob, kind: "processed" };
    }

    return outcome;
  }
}
