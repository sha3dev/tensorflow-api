/**
 * @section imports:externals
 */

import { createAdaptorServer } from "@hono/node-server";
import type { ServerType } from "@hono/node-server";
import { Hono } from "hono";
import type { Context } from "hono";
import type { ContentfulStatusCode } from "hono/utils/http-status";

/**
 * @section imports:internals
 */

import type { AppInfoService } from "../app-info/index.ts";
import config from "../config.ts";
import type { DashboardStateService } from "../dashboard-state/index.ts";
import type { CreatePredictionJobRequest, CreateTrainingJobRequest, JobListFilter, JobStatus } from "../job/index.ts";
import type { JobService } from "../job/index.ts";
import type { CreateModelRequest, KerasModelDefinition } from "../model/index.ts";
import type { ModelService } from "../model/index.ts";

/**
 * @section class
 */

export class HttpServerService {
  /**
   * @section private:attributes
   */

  private readonly appInfoService: AppInfoService;

  private readonly dashboardStateService: DashboardStateService;

  private readonly jobService: JobService;

  private readonly modelService: ModelService;

  /**
   * @section constructor
   */

  public constructor(appInfoService: AppInfoService, modelService: ModelService, jobService: JobService, dashboardStateService: DashboardStateService) {
    this.appInfoService = appInfoService;
    this.modelService = modelService;
    this.jobService = jobService;
    this.dashboardStateService = dashboardStateService;
  }

  /**
   * @section factory
   */

  public static create(
    appInfoService: AppInfoService,
    modelService: ModelService,
    jobService: JobService,
    dashboardStateService: DashboardStateService,
  ): HttpServerService {
    const service = new HttpServerService(appInfoService, modelService, jobService, dashboardStateService);
    return service;
  }

  /**
   * @section private:methods
   */

  private isRecord(payload: unknown): payload is Record<string, unknown> {
    const isPayloadRecord = Boolean(payload) && typeof payload === "object" && !Array.isArray(payload);
    return isPayloadRecord;
  }

  private buildJsonError(code: string, message: string): { code: string; message: string } {
    const payload = { code, message };
    return payload;
  }

  private buildDashboardHtml(): string {
    const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>TensorFlow API Dashboard</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f4f6f9;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --line: #d8dee8;
        --line-strong: #c3ccd8;
        --ink: #16202f;
        --muted: #5d6b82;
        --accent: #0f5bd8;
        --success: #0f8a5f;
        --success-soft: rgba(15, 138, 95, 0.12);
        --danger: #c03f2f;
        --danger-soft: rgba(192, 63, 47, 0.12);
        --warning: #a26400;
        --warning-soft: rgba(162, 100, 0, 0.12);
        --shadow: 0 8px 24px rgba(18, 30, 52, 0.06);
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Inter", "Segoe UI", sans-serif;
        background: linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
        color: var(--ink);
      }
      main {
        max-width: 1460px;
        margin: 0 auto;
        padding: 16px 16px 28px;
      }
      a {
        color: inherit;
      }
      h1,
      h2 {
        margin: 0;
      }
      .shell {
        display: grid;
        gap: 12px;
      }
      .toolbar {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 10px 16px;
        align-items: center;
        padding: 12px 14px;
        border-radius: 14px;
        background: var(--surface-soft);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
      }
      .toolbar-title {
        display: grid;
        gap: 2px;
      }
      .toolbar-title h1 {
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
      }
      .toolbar-note {
        color: var(--muted);
        font-size: 0.85rem;
      }
      .cards {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }
      .card,
      .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
      }
      .card {
        padding: 12px 14px;
        box-shadow: var(--shadow);
      }
      .card-label {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      .card strong {
        display: block;
        margin-top: 8px;
        font-size: 2rem;
        letter-spacing: -0.04em;
      }
      .card-foot {
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.8rem;
      }
      .layout {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
      }
      .panel {
        overflow: hidden;
        box-shadow: var(--shadow);
      }
      .panel-header {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        align-items: center;
        padding: 12px 14px 10px;
        border-bottom: 1px solid var(--line);
        background: var(--surface-soft);
      }
      .panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: -0.01em;
      }
      .panel-note {
        color: var(--muted);
        font-size: 0.8rem;
      }
      .table-wrap {
        overflow: auto;
        max-height: calc(100vh - 230px);
      }
      table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
      }
      th,
      td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
        white-space: nowrap;
      }
      th {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: var(--surface-soft);
        position: sticky;
        top: 0;
        z-index: 1;
      }
      tbody tr:nth-child(even) {
        background: #fbfcfe;
      }
      tbody tr:hover {
        background: #f3f7fd;
      }
      button {
        border: 0;
        border-radius: 10px;
        padding: 9px 14px;
        background: var(--accent);
        color: #fff;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
        box-shadow: none;
      }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
      }
      .status-ready,
      .status-succeeded {
        color: var(--success);
      }
      .status-pending,
      .status-queued,
      .status-running {
        color: var(--warning);
      }
      .status-failed {
        color: var(--danger);
      }
      .pill.status-ready,
      .pill.status-succeeded {
        background: var(--success-soft);
      }
      .pill.status-pending,
      .pill.status-queued,
      .pill.status-running {
        background: var(--warning-soft);
      }
      .pill.status-failed {
        background: var(--danger-soft);
      }
      .model-link,
      .job-link {
        color: var(--ink);
        font-weight: 700;
        text-decoration: none;
      }
      .model-link:hover,
      .job-link:hover {
        color: var(--accent);
      }
      .secondary {
        display: block;
        margin-top: 2px;
        color: var(--muted);
        font-size: 0.78rem;
      }
      .mono {
        font-family: "SFMono-Regular", "Menlo", monospace;
        font-size: 0.76rem;
      }
      .feedback {
        min-height: 18px;
        color: var(--muted);
        font-size: 0.82rem;
        text-align: right;
      }
      .empty {
        padding: 18px 12px;
        color: var(--muted);
        text-align: center;
      }
      .muted {
        color: var(--muted);
      }
      .table-compact td,
      .table-compact th {
        padding-right: 10px;
        padding-left: 10px;
      }
      .table-main td:nth-child(1),
      .table-main th:nth-child(1),
      .table-jobs td:nth-child(1),
      .table-jobs th:nth-child(1),
      .table-jobs td:nth-child(8),
      .table-jobs th:nth-child(8) {
        white-space: normal;
      }
      @media (max-width: 980px) {
        .layout {
          grid-template-columns: 1fr;
        }
        .cards {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .table-wrap {
          max-height: none;
        }
      }
      @media (max-width: 720px) {
        main {
          padding: 12px 10px 20px;
        }
        .toolbar {
          grid-template-columns: 1fr;
        }
        .feedback {
          text-align: left;
        }
        .cards {
          grid-template-columns: 1fr 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <div class="shell">
      <div class="toolbar">
        <div class="toolbar-title">
          <h1>TensorFlow API dashboard</h1>
          <div class="toolbar-note">Live state from <span class="mono">/api/state</span>. Auto-refresh every 5 seconds.</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
          <span id="feedback" class="feedback">Loading state…</span>
          <button id="refresh-button" type="button">Refresh</button>
        </div>
      </div>
      <section id="summary-cards" class="cards"></section>
      <div class="layout">
      <section class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">Models</h2>
            <div class="panel-note">Creation time, current status, and recent training or prediction activity.</div>
          </div>
        </div>
        <div class="table-wrap">
        <table class="table-main">
          <thead>
            <tr>
              <th>Model</th>
              <th>Status</th>
              <th>Created</th>
              <th>Train Count</th>
              <th>Predict Count</th>
              <th>Last Train</th>
              <th>Last Predict</th>
            </tr>
          </thead>
          <tbody id="models-body"></tbody>
        </table>
        </div>
      </section>
      <section class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">Recent Jobs</h2>
            <div class="panel-note">Most recent create, train, and predict jobs with error visibility.</div>
          </div>
        </div>
        <div class="table-wrap">
        <table class="table-compact table-jobs">
          <thead>
            <tr>
              <th>Job</th>
              <th>Type</th>
              <th>Model</th>
              <th>Status</th>
              <th>Created</th>
              <th>Started</th>
              <th>Finished</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody id="jobs-body"></tbody>
        </table>
        </div>
      </section>
      </div>
      </div>
    </main>
    <script src="/dashboard/app.js" type="module"></script>
  </body>
</html>`;
    return html;
  }

  private buildDashboardScript(): string {
    const script = `const stateUrl = "/api/state";

const feedbackElement = document.getElementById("feedback");
const summaryElement = document.getElementById("summary-cards");
const modelsElement = document.getElementById("models-body");
const jobsElement = document.getElementById("jobs-body");
const refreshButton = document.getElementById("refresh-button");
const dateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

const formatDate = (value) => {
  if (!value) {
    return "No activity yet";
  }

  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return value;
  }

  return dateTimeFormatter.format(timestamp);
};

const escapeHtml = (value) => {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
};

const renderStatusPill = (status) => {
  return '<span class="pill status-' + escapeHtml(status) + '">' + escapeHtml(status.replaceAll("_", " ")) + "</span>";
};

const renderSummary = (summary) => {
  summaryElement.innerHTML = [
    ["Models", summary.modelCount, "Registered model records currently available."],
    ["Queued Jobs", summary.queuedJobCount, "Waiting for the local Python worker loop."],
    ["Running Jobs", summary.runningJobCount, "Currently executing TensorFlow work."],
    ["Failed Jobs", summary.failedJobCount, "Recent jobs that require inspection."],
  ].map(([label, value, note]) => {
    return '<article class="card"><span class="card-label">' + label + '</span><strong>' + value + "</strong><div class='card-foot'>" + note + "</div></article>";
  }).join("");
};

const renderModels = (models) => {
  if (models.length === 0) {
    modelsElement.innerHTML = "<tr><td class='empty' colspan='7'>No models have been created yet.</td></tr>";
    return;
  }

  modelsElement.innerHTML = models.map((model) => {
    return "<tr>" +
      "<td><a class='model-link' href='/api/models/" + encodeURIComponent(model.modelId) + "' target='_blank' rel='noreferrer'>" + escapeHtml(model.modelId) + "</a><span class='secondary mono'>" + escapeHtml(model.definitionPath) + "</span></td>" +
      "<td>" + renderStatusPill(model.status) + "</td>" +
      "<td>" + formatDate(model.createdAt) + "</td>" +
      "<td>" + escapeHtml(model.trainingCount) + "</td>" +
      "<td>" + escapeHtml(model.predictionCount) + "</td>" +
      "<td>" + formatDate(model.lastTrainingAt) + "</td>" +
      "<td>" + formatDate(model.lastPredictionAt) + "</td>" +
      "</tr>";
  }).join("");
};

const renderJobs = (jobs) => {
  if (jobs.length === 0) {
    jobsElement.innerHTML = "<tr><td class='empty' colspan='8'>No jobs have been recorded yet.</td></tr>";
    return;
  }

  jobsElement.innerHTML = jobs.map((job) => {
    const errorText = job.errorMessage ? escapeHtml(job.errorMessage) : "<span class='muted'>No error</span>";
    return "<tr>" +
      "<td><a class='job-link mono' href='/api/jobs/" + encodeURIComponent(job.jobId) + "' target='_blank' rel='noreferrer'>" + escapeHtml(job.jobId) + "</a></td>" +
      "<td>" + escapeHtml(job.jobType.replaceAll("_", " ")) + "</td>" +
      "<td>" + escapeHtml(job.modelId) + "</td>" +
      "<td>" + renderStatusPill(job.status) + "</td>" +
      "<td>" + formatDate(job.createdAt) + "</td>" +
      "<td>" + formatDate(job.startedAt) + "</td>" +
      "<td>" + formatDate(job.finishedAt) + "</td>" +
      "<td>" + errorText + "</td>" +
      "</tr>";
  }).join("");
};

const loadState = async () => {
  feedbackElement.textContent = "Loading state…";
  try {
    const response = await fetch(stateUrl);
    if (!response.ok) {
      throw new Error("dashboard state request failed with status " + response.status);
    }
    const payload = await response.json();
    renderSummary(payload.summary);
    renderModels(payload.models);
    renderJobs(payload.recentJobs);
    feedbackElement.textContent = "Last refresh: " + formatDate(new Date().toISOString());
  } catch (error) {
    feedbackElement.textContent = error instanceof Error ? error.message : "Unknown dashboard error";
  }
};

refreshButton?.addEventListener("click", () => {
  void loadState();
});

void loadState();
setInterval(() => {
  void loadState();
}, 5000);
`;
    return script;
  }

  private isSafeModelId(modelId: string): boolean {
    const isSafe = /^[a-zA-Z0-9_-]+$/.test(modelId);
    return isSafe;
  }

  private async readJsonBody(context: Context): Promise<Record<string, unknown>> {
    const parsedBody = (await context.req.json()) as unknown;
    let payload: Record<string, unknown>;

    if (this.isRecord(parsedBody)) {
      payload = parsedBody;
    } else {
      throw new Error("invalid_request: request body must be a JSON object");
    }

    return payload;
  }

  private createJsonResponse(context: Context, payload: unknown, statusCode: number): Response {
    context.header("content-type", config.RESPONSE_CONTENT_TYPE);
    const response = context.json(payload, statusCode as ContentfulStatusCode);
    return response;
  }

  private requirePathParam(context: Context, parameterName: string): string {
    const parameterValue = context.req.param(parameterName);

    if (!parameterValue) {
      throw new Error(`invalid_request: missing path parameter '${parameterName}'`);
    }

    return parameterValue;
  }

  private handleRootRequest(context: Context): Response {
    const payload = this.appInfoService.buildPayload();
    const response = this.createJsonResponse(context, payload, 200);
    return response;
  }

  private handleDashboardRequest(context: Context): Response {
    const response = context.html(this.buildDashboardHtml(), 200);
    return response;
  }

  private handleDashboardScriptRequest(context: Context): Response {
    context.header("content-type", "text/javascript; charset=utf-8");
    const response = context.body(this.buildDashboardScript(), 200);
    return response;
  }

  private handleStateRequest(context: Context): Response {
    const payload = this.dashboardStateService.buildState();
    const response = this.createJsonResponse(context, payload, 200);
    return response;
  }

  private handleListModelsRequest(context: Context): Response {
    const payload = this.modelService.listModels();
    const response = this.createJsonResponse(context, payload, 200);
    return response;
  }

  private handleGetModelRequest(context: Context): Response {
    const modelId = this.requirePathParam(context, "modelId");
    const modelRecord = this.modelService.getModel(modelId);
    let response: Response;

    if (!modelRecord) {
      response = this.createJsonResponse(context, this.buildJsonError("not_found", `model '${modelId}' was not found`), 404);
    } else {
      response = this.createJsonResponse(context, modelRecord, 200);
    }

    return response;
  }

  private buildCreateModelRequest(payload: Record<string, unknown>): CreateModelRequest {
    const modelId = typeof payload.modelId === "string" ? payload.modelId : "";
    const definition = payload.definition;

    if (!modelId.trim() || !this.isSafeModelId(modelId)) {
      throw new Error("invalid_request: modelId must be a safe non-empty identifier");
    }

    if (!this.isRecord(definition) || typeof definition.format !== "string" || !this.isRecord(definition.modelConfig)) {
      throw new Error("invalid_request: definition must contain format and modelConfig");
    }

    if (definition.format !== "keras-sequential" && definition.format !== "keras-functional") {
      throw new Error("invalid_request: definition.format must be keras-sequential or keras-functional");
    }

    const request: CreateModelRequest = {
      definition: definition as KerasModelDefinition,
      modelId,
    };
    return request;
  }

  private async handleCreateModelRequest(context: Context): Promise<Response> {
    const payload = await this.readJsonBody(context);
    const createRequest = this.buildCreateModelRequest(payload);
    const createResult = this.modelService.createModel(createRequest);
    let response: Response;

    if (createResult.kind === "conflict") {
      response = this.createJsonResponse(context, this.buildJsonError("conflict", createResult.message), 409);
    } else {
      context.header("Location", `/api/models/${encodeURIComponent(createResult.model.modelId)}`);
      response = this.createJsonResponse(context, createResult, 201);
    }

    return response;
  }

  private buildTrainingJobRequest(modelId: string, payload: Record<string, unknown>): CreateTrainingJobRequest {
    if (!this.isSafeModelId(modelId)) {
      throw new Error("invalid_request: modelId must be a safe identifier");
    }

    if (!this.isRecord(payload.trainingInput)) {
      throw new Error("invalid_request: trainingInput is required");
    }

    const request: CreateTrainingJobRequest = this.isRecord(payload.fitConfig)
      ? {
          fitConfig: payload.fitConfig as NonNullable<CreateTrainingJobRequest["fitConfig"]>,
          trainingInput: payload.trainingInput as CreateTrainingJobRequest["trainingInput"],
        }
      : {
          trainingInput: payload.trainingInput as CreateTrainingJobRequest["trainingInput"],
        };
    return request;
  }

  private async handleCreateTrainingJobRequest(context: Context): Promise<Response> {
    const modelId = this.requirePathParam(context, "modelId");
    const payload = await this.readJsonBody(context);
    const createRequest = this.buildTrainingJobRequest(modelId, payload);
    const createResult = this.jobService.enqueueTrainingJob(modelId, createRequest);
    let response: Response;

    if (createResult.kind === "created") {
      response = this.createJsonResponse(context, createResult.job, 202);
    } else {
      if (createResult.kind === "not_found") {
        response = this.createJsonResponse(context, this.buildJsonError("not_found", createResult.message), 404);
      } else {
        response = this.createJsonResponse(context, this.buildJsonError("conflict", createResult.message), 409);
      }
    }

    return response;
  }

  private buildPredictionJobRequest(modelId: string, payload: Record<string, unknown>): CreatePredictionJobRequest {
    if (!this.isSafeModelId(modelId)) {
      throw new Error("invalid_request: modelId must be a safe identifier");
    }

    if (!this.isRecord(payload.predictionInput)) {
      throw new Error("invalid_request: predictionInput is required");
    }

    const request: CreatePredictionJobRequest = {
      predictionInput: payload.predictionInput as CreatePredictionJobRequest["predictionInput"],
    };
    return request;
  }

  private async handleCreatePredictionJobRequest(context: Context): Promise<Response> {
    const modelId = this.requirePathParam(context, "modelId");
    const payload = await this.readJsonBody(context);
    const createRequest = this.buildPredictionJobRequest(modelId, payload);
    const createResult = this.jobService.enqueuePredictionJob(modelId, createRequest);
    let response: Response;

    if (createResult.kind === "created") {
      response = this.createJsonResponse(context, createResult.job, 202);
    } else {
      if (createResult.kind === "not_found") {
        response = this.createJsonResponse(context, this.buildJsonError("not_found", createResult.message), 404);
      } else {
        response = this.createJsonResponse(context, this.buildJsonError("conflict", createResult.message), 409);
      }
    }

    return response;
  }

  private buildJobListFilter(context: Context): JobListFilter | undefined {
    const modelId = context.req.query("modelId");
    const status = context.req.query("status");
    const normalizedStatus: JobStatus | null = status === "queued" || status === "running" || status === "succeeded" || status === "failed" ? status : null;
    const filter: JobListFilter | undefined =
      modelId || normalizedStatus
        ? {
            ...(modelId ? { modelId } : {}),
            ...(normalizedStatus ? { status: normalizedStatus } : {}),
          }
        : undefined;
    return filter;
  }

  private handleListJobsRequest(context: Context): Response {
    const payload = this.jobService.listJobs(this.buildJobListFilter(context));
    const response = this.createJsonResponse(context, payload, 200);
    return response;
  }

  private handleGetJobRequest(context: Context): Response {
    const jobId = this.requirePathParam(context, "jobId");
    const jobRecord = this.jobService.getJob(jobId);
    let response: Response;

    if (!jobRecord) {
      response = this.createJsonResponse(context, this.buildJsonError("not_found", `job '${jobId}' was not found`), 404);
    } else {
      response = this.createJsonResponse(context, jobRecord, 200);
    }

    return response;
  }

  private handleGetJobResultRequest(context: Context): Response {
    const jobId = this.requirePathParam(context, "jobId");
    const jobResult = this.jobService.getJobResult(jobId);
    let response: Response;

    if (jobResult.kind === "not_found") {
      response = this.createJsonResponse(context, this.buildJsonError("not_found", jobResult.message), 404);
    } else {
      if (jobResult.kind === "not_ready") {
        response = this.createJsonResponse(context, this.buildJsonError("conflict", `job '${jobId}' is not finished yet`), 409);
      } else {
        response = this.createJsonResponse(context, jobResult.result, 200);
      }
    }

    return response;
  }

  /**
   * @section public:methods
   */

  public buildServer(): ServerType {
    const app = new Hono();
    app.onError((error, context) => {
      const isInvalidRequest = error.message.startsWith("invalid_request:");
      const response = this.createJsonResponse(
        context,
        this.buildJsonError(
          isInvalidRequest ? "invalid_request" : "internal_error",
          isInvalidRequest ? error.message.replace("invalid_request: ", "") : error.message || "internal server error",
        ),
        isInvalidRequest ? 400 : 500,
      );
      return response;
    });
    app.get("/", this.handleRootRequest.bind(this));
    app.get("/dashboard", this.handleDashboardRequest.bind(this));
    app.get("/dashboard/app.js", this.handleDashboardScriptRequest.bind(this));
    app.get("/api/state", this.handleStateRequest.bind(this));
    app.get("/api/models", this.handleListModelsRequest.bind(this));
    app.get("/api/models/:modelId", this.handleGetModelRequest.bind(this));
    app.post("/api/models", this.handleCreateModelRequest.bind(this));
    app.post("/api/models/:modelId/training-jobs", this.handleCreateTrainingJobRequest.bind(this));
    app.post("/api/models/:modelId/prediction-jobs", this.handleCreatePredictionJobRequest.bind(this));
    app.get("/api/jobs", this.handleListJobsRequest.bind(this));
    app.get("/api/jobs/:jobId", this.handleGetJobRequest.bind(this));
    app.get("/api/jobs/:jobId/result", this.handleGetJobResultRequest.bind(this));
    return createAdaptorServer({ fetch: app.fetch });
  }
}
