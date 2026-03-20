/**
 * @section imports:externals
 */

import Database from "better-sqlite3";

/**
 * @section imports:internals
 */

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import config from "../config.ts";
import type { JobListFilter, JobRecord, JobResultPayload, JobStatus, JobType } from "../job/index.ts";
import type { ModelRecord, ModelStatus } from "../model/index.ts";

/**
 * @section types
 */

type InsertModelRecordCommand = {
  artifactPath: string;
  createdAt: string;
  definitionPath: string;
  metadata: Record<string, unknown> | null;
  modelId: string;
  status: ModelStatus;
  updatedAt: string;
};

type InsertJobRecordCommand = {
  createdAt: string;
  jobId: string;
  jobType: JobType;
  modelId: string;
  requestPath: string;
  resultPath: string;
  startedAt: string | null;
  status: JobStatus;
};

type JobStatusSummary = {
  failedJobCount: number;
  queuedJobCount: number;
  runningJobCount: number;
};

type ModelDeletionPlan = {
  jobDirectoryPaths: string[];
  modelDirectoryPath: string;
};

/**
 * @section class
 */

export class StorageService {
  /**
   * @section private:attributes
   */

  private readonly database: Database.Database;

  private readonly storageRoot: string;

  /**
   * @section constructor
   */

  public constructor(databasePath: string, storageRoot: string) {
    this.storageRoot = storageRoot;
    mkdirSync(dirname(databasePath), { recursive: true });
    this.database = new Database(databasePath);
    this.database.pragma("journal_mode = WAL");
  }

  /**
   * @section factory
   */

  public static createDefault(): StorageService {
    const service = new StorageService(config.SQLITE_PATH, config.STORAGE_ROOT);
    return service;
  }

  /**
   * @section private:methods
   */

  private ensureDirectoryExists(targetPath: string): void {
    mkdirSync(targetPath, { recursive: true });
  }

  private ensureParentDirectoryExists(targetPath: string): void {
    this.ensureDirectoryExists(dirname(targetPath));
  }

  private hasTableColumn(tableName: string, columnName: string): boolean {
    const rows = this.database.prepare(`PRAGMA table_info(${tableName})`).all() as Record<string, unknown>[];
    const hasColumn = rows.some((row) => {
      return row.name === columnName;
    });
    return hasColumn;
  }

  private parseModelMetadata(metadataValue: unknown): Record<string, unknown> | null {
    let metadata: Record<string, unknown> | null = null;

    if (typeof metadataValue === "string") {
      const parsedValue = JSON.parse(metadataValue) as unknown;

      if (parsedValue && typeof parsedValue === "object" && !Array.isArray(parsedValue)) {
        metadata = parsedValue as Record<string, unknown>;
      }
    }

    return metadata;
  }

  private buildModelDeletionPlan(modelId: string): ModelDeletionPlan {
    const jobRows = this.database.prepare("SELECT job_id FROM job_record WHERE model_id = ?").all(modelId) as Record<string, unknown>[];
    const jobDirectoryPaths = jobRows.map((row) => {
      return this.getJobDirectory(String(row.job_id));
    });
    const deletionPlan: ModelDeletionPlan = {
      jobDirectoryPaths,
      modelDirectoryPath: this.getModelDirectory(modelId),
    };
    return deletionPlan;
  }

  private toModelRecord(row: Record<string, unknown>): ModelRecord {
    const modelRecord: ModelRecord = {
      artifactPath: String(row.artifact_path),
      createdAt: String(row.created_at),
      definitionPath: String(row.definition_path),
      lastTrainingAt: row.last_training_at ? String(row.last_training_at) : null,
      lastTrainingJobId: row.last_training_job_id ? String(row.last_training_job_id) : null,
      metadata: this.parseModelMetadata(row.metadata_json),
      modelId: String(row.model_id),
      status: row.status as ModelStatus,
      trainingCount: Number(row.training_count),
      updatedAt: String(row.updated_at),
    };
    return modelRecord;
  }

  private toJobRecord(row: Record<string, unknown>): JobRecord {
    const jobRecord: JobRecord = {
      createdAt: String(row.created_at),
      errorCode: row.error_code ? String(row.error_code) : null,
      errorMessage: row.error_message ? String(row.error_message) : null,
      finishedAt: row.finished_at ? String(row.finished_at) : null,
      jobId: String(row.job_id),
      jobType: row.job_type as JobType,
      modelId: String(row.model_id),
      requestPath: String(row.request_path),
      resultPath: String(row.result_path),
      startedAt: row.started_at ? String(row.started_at) : null,
      status: row.status as JobStatus,
    };
    return jobRecord;
  }

  /**
   * @section public:methods
   */

  public initialize(): void {
    this.ensureDirectoryExists(this.storageRoot);
    this.ensureDirectoryExists(this.getModelDirectoryRoot());
    this.ensureDirectoryExists(this.getJobDirectoryRoot());
    this.database.exec(`
      CREATE TABLE IF NOT EXISTS model_record (
        model_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL,
        training_count INTEGER NOT NULL,
        prediction_count INTEGER NOT NULL,
        last_training_at TEXT,
        last_prediction_at TEXT,
        last_training_job_id TEXT,
        last_prediction_job_id TEXT,
        metadata_json TEXT,
        definition_path TEXT NOT NULL,
        artifact_path TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS job_record (
        job_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        error_code TEXT,
        error_message TEXT,
        request_path TEXT NOT NULL,
        result_path TEXT NOT NULL
      );
    `);

    if (!this.hasTableColumn("model_record", "metadata_json")) {
      this.database.exec("ALTER TABLE model_record ADD COLUMN metadata_json TEXT");
    }
  }

  public close(): void {
    this.database.close();
  }

  public getModelDirectoryRoot(): string {
    const modelDirectoryRoot = join(this.storageRoot, "models");
    return modelDirectoryRoot;
  }

  public getJobDirectoryRoot(): string {
    const jobDirectoryRoot = join(this.storageRoot, "jobs");
    return jobDirectoryRoot;
  }

  public getModelDirectory(modelId: string): string {
    const modelDirectory = join(this.getModelDirectoryRoot(), modelId);
    return modelDirectory;
  }

  public getJobDirectory(jobId: string): string {
    const jobDirectory = join(this.getJobDirectoryRoot(), jobId);
    return jobDirectory;
  }

  public ensureModelDirectory(modelId: string): string {
    const modelDirectory = this.getModelDirectory(modelId);
    this.ensureDirectoryExists(modelDirectory);
    return modelDirectory;
  }

  public ensureJobDirectory(jobId: string): string {
    const jobDirectory = this.getJobDirectory(jobId);
    this.ensureDirectoryExists(jobDirectory);
    return jobDirectory;
  }

  public writeJsonFile(targetPath: string, payload: unknown): void {
    this.ensureParentDirectoryExists(targetPath);
    writeFileSync(targetPath, JSON.stringify(payload, null, 2), "utf8");
  }

  public readJsonFile(targetPath: string): unknown {
    const content = readFileSync(targetPath, "utf8");
    const payload = JSON.parse(content) as unknown;
    return payload;
  }

  public hasFile(targetPath: string): boolean {
    const hasFile = existsSync(targetPath);
    return hasFile;
  }

  public deleteDirectoryIfPresent(targetPath: string): void {
    if (this.hasFile(targetPath) || existsSync(targetPath)) {
      rmSync(targetPath, { force: true, recursive: true });
    }
  }

  public insertModelRecord(command: InsertModelRecordCommand): boolean {
    const executionResult = this.database
      .prepare(`
        INSERT OR IGNORE INTO model_record (
          model_id,
          created_at,
          updated_at,
          status,
          training_count,
          prediction_count,
          last_training_at,
          last_prediction_at,
          last_training_job_id,
          last_prediction_job_id,
          metadata_json,
          definition_path,
          artifact_path
        ) VALUES (?, ?, ?, ?, 0, 0, NULL, NULL, NULL, NULL, ?, ?, ?)
      `)
      .run(
        command.modelId,
        command.createdAt,
        command.updatedAt,
        command.status,
        command.metadata ? JSON.stringify(command.metadata) : null,
        command.definitionPath,
        command.artifactPath,
      );
    const hasInsertedRecord = executionResult.changes > 0;
    return hasInsertedRecord;
  }

  public getModelRecord(modelId: string): ModelRecord | null {
    const row = this.database.prepare("SELECT * FROM model_record WHERE model_id = ?").get(modelId) as Record<string, unknown> | undefined;
    let modelRecord: ModelRecord | null = null;

    if (row) {
      modelRecord = this.toModelRecord(row);
    }

    return modelRecord;
  }

  public listModelRecords(): ModelRecord[] {
    const rows = this.database.prepare("SELECT * FROM model_record ORDER BY created_at DESC, model_id ASC").all() as Record<string, unknown>[];
    const modelRecords = rows.map((row) => {
      return this.toModelRecord(row);
    });
    return modelRecords;
  }

  public markModelReady(modelId: string, updatedAt: string): void {
    this.database.prepare("UPDATE model_record SET status = ?, updated_at = ? WHERE model_id = ?").run("ready", updatedAt, modelId);
  }

  public markModelFailed(modelId: string, updatedAt: string): void {
    this.database.prepare("UPDATE model_record SET status = ?, updated_at = ? WHERE model_id = ?").run("failed", updatedAt, modelId);
  }

  public markModelTrainingQueued(modelId: string, jobId: string, updatedAt: string): void {
    this.database.prepare("UPDATE model_record SET last_training_job_id = ?, updated_at = ? WHERE model_id = ?").run(jobId, updatedAt, modelId);
  }

  public updateModelMetadata(modelId: string, metadata: Record<string, unknown>, updatedAt: string): boolean {
    const executionResult = this.database
      .prepare("UPDATE model_record SET metadata_json = ?, updated_at = ? WHERE model_id = ?")
      .run(JSON.stringify(metadata), updatedAt, modelId);
    const hasUpdatedModel = executionResult.changes > 0;
    return hasUpdatedModel;
  }

  public markModelTrainingSucceeded(modelId: string, jobId: string, updatedAt: string): void {
    this.database
      .prepare(`
        UPDATE model_record
        SET training_count = training_count + 1,
            last_training_at = ?,
            last_training_job_id = ?,
            updated_at = ?
        WHERE model_id = ?
      `)
      .run(updatedAt, jobId, updatedAt, modelId);
  }

  public completeTrainingJob(jobId: string, modelId: string, finishedAt: string, modelMetadata?: Record<string, unknown>): void {
    const transaction = this.database.transaction(
      (trainingJobId: string, trainingModelId: string, completedAt: string, metadata: Record<string, unknown> | undefined) => {
        this.database
          .prepare("UPDATE job_record SET status = ?, finished_at = ?, error_code = NULL, error_message = NULL WHERE job_id = ?")
          .run("succeeded", completedAt, trainingJobId);

        if (metadata) {
          this.database
            .prepare(`
            UPDATE model_record
            SET training_count = training_count + 1,
                last_training_at = ?,
                last_training_job_id = ?,
                metadata_json = ?,
                updated_at = ?
            WHERE model_id = ?
          `)
            .run(completedAt, trainingJobId, JSON.stringify(metadata), completedAt, trainingModelId);
        } else {
          this.database
            .prepare(`
            UPDATE model_record
            SET training_count = training_count + 1,
                last_training_at = ?,
                last_training_job_id = ?,
                updated_at = ?
            WHERE model_id = ?
          `)
            .run(completedAt, trainingJobId, completedAt, trainingModelId);
        }
      },
    );

    transaction(jobId, modelId, finishedAt, modelMetadata);
  }

  public hasActiveJobsForModel(modelId: string): boolean {
    const row = this.database
      .prepare("SELECT COUNT(*) AS active_job_count FROM job_record WHERE model_id = ? AND status IN (?, ?)")
      .get(modelId, "queued", "running") as Record<string, unknown>;
    const hasActiveJobs = Number(row.active_job_count) > 0;
    return hasActiveJobs;
  }

  public deleteModel(modelId: string): boolean {
    const modelRecord = this.getModelRecord(modelId);
    let hasDeletedModel = false;

    if (modelRecord) {
      const deletionPlan = this.buildModelDeletionPlan(modelId);
      const transaction = this.database.transaction((deletedModelId: string) => {
        this.database.prepare("DELETE FROM job_record WHERE model_id = ?").run(deletedModelId);
        this.database.prepare("DELETE FROM model_record WHERE model_id = ?").run(deletedModelId);
      });

      transaction(modelId);

      for (const jobDirectoryPath of deletionPlan.jobDirectoryPaths) {
        this.deleteDirectoryIfPresent(jobDirectoryPath);
      }

      this.deleteDirectoryIfPresent(deletionPlan.modelDirectoryPath);
      hasDeletedModel = true;
    }

    return hasDeletedModel;
  }

  public insertJobRecord(command: InsertJobRecordCommand): void {
    this.database
      .prepare(`
        INSERT INTO job_record (
          job_id,
          model_id,
          job_type,
          status,
          created_at,
          started_at,
          finished_at,
          error_code,
          error_message,
          request_path,
          result_path
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
      `)
      .run(command.jobId, command.modelId, command.jobType, command.status, command.createdAt, command.startedAt, command.requestPath, command.resultPath);
  }

  public getJobRecord(jobId: string): JobRecord | null {
    const row = this.database.prepare("SELECT * FROM job_record WHERE job_id = ?").get(jobId) as Record<string, unknown> | undefined;
    let jobRecord: JobRecord | null = null;

    if (row) {
      jobRecord = this.toJobRecord(row);
    }

    return jobRecord;
  }

  public listJobRecords(filter?: JobListFilter): JobRecord[] {
    const clauses: string[] = [];
    const values: Array<string | number> = [];
    let sql = "SELECT * FROM job_record";

    if (filter?.modelId) {
      clauses.push("model_id = ?");
      values.push(filter.modelId);
    }

    if (filter?.status) {
      clauses.push("status = ?");
      values.push(filter.status);
    }

    if (clauses.length > 0) {
      sql += ` WHERE ${clauses.join(" AND ")}`;
    }

    sql += " ORDER BY created_at DESC, job_id ASC";

    if (filter?.limit) {
      sql += " LIMIT ?";
      values.push(filter.limit);
    }

    const rows = this.database.prepare(sql).all(...values) as Record<string, unknown>[];
    const jobRecords = rows.map((row) => {
      return this.toJobRecord(row);
    });
    return jobRecords;
  }

  public countJobStatuses(): JobStatusSummary {
    const rows = this.database.prepare("SELECT status, COUNT(*) AS total_count FROM job_record GROUP BY status").all() as Record<string, unknown>[];
    let queuedJobCount = 0;
    let runningJobCount = 0;
    let failedJobCount = 0;

    for (const row of rows) {
      if (row.status === "queued") {
        queuedJobCount = Number(row.total_count);
      }

      if (row.status === "running") {
        runningJobCount = Number(row.total_count);
      }

      if (row.status === "failed") {
        failedJobCount = Number(row.total_count);
      }
    }

    return { failedJobCount, queuedJobCount, runningJobCount };
  }

  public markRunningJobsFailedAfterRestart(updatedAt: string): void {
    this.database
      .prepare(`
        UPDATE job_record
        SET status = ?,
            finished_at = ?,
            error_code = ?,
            error_message = ?
        WHERE status = ?
      `)
      .run("failed", updatedAt, "internal_error", "job was interrupted by a service restart", "running");
  }

  public claimNextQueuedJob(startedAt: string): JobRecord | null {
    const transaction = this.database.transaction((startedAtValue: string) => {
      const row = this.database.prepare("SELECT * FROM job_record WHERE status = ? ORDER BY created_at ASC, job_id ASC LIMIT 1").get("queued") as
        | Record<string, unknown>
        | undefined;
      let claimedJob: JobRecord | null = null;

      if (row) {
        this.database.prepare("UPDATE job_record SET status = ?, started_at = ? WHERE job_id = ?").run("running", startedAtValue, row.job_id);
        claimedJob = this.getJobRecord(String(row.job_id));
      }

      return claimedJob;
    });
    const claimedJob = transaction(startedAt);
    return claimedJob;
  }

  public markJobSucceeded(jobId: string, finishedAt: string): void {
    this.database
      .prepare("UPDATE job_record SET status = ?, finished_at = ?, error_code = NULL, error_message = NULL WHERE job_id = ?")
      .run("succeeded", finishedAt, jobId);
  }

  public markJobFailed(jobId: string, finishedAt: string, errorCode: string, errorMessage: string): void {
    this.database
      .prepare("UPDATE job_record SET status = ?, finished_at = ?, error_code = ?, error_message = ? WHERE job_id = ?")
      .run("failed", finishedAt, errorCode, errorMessage, jobId);
  }

  public readJobResult(resultPath: string): JobResultPayload | null {
    let resultPayload: JobResultPayload | null = null;

    if (this.hasFile(resultPath)) {
      resultPayload = this.readJsonFile(resultPath) as JobResultPayload;
    }

    return resultPayload;
  }
}
