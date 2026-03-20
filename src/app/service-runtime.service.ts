import type { ServerType } from "@hono/node-server";

/**
 * @section imports:internals
 */

import { AppInfoService } from "../app-info/index.ts";
import config from "../config.ts";
import { DashboardStateService } from "../dashboard-state/index.ts";
import { HttpServerService } from "../http/index.ts";
import { JobService } from "../job/index.ts";
import logger from "../logger.ts";
import { ModelService } from "../model/index.ts";
import { PythonRuntimeService } from "../python-runtime/index.ts";
import { StorageService } from "../storage/index.ts";

/**
 * @section class
 */

export class ServiceRuntime {
  /**
   * @section private:attributes
   */

  private readonly httpServerService: HttpServerService;

  private readonly jobPollIntervalMs: number;

  private readonly jobService: JobService;

  private readonly pythonRuntimeService: PythonRuntimeService;

  private intervalReference: NodeJS.Timeout | null;

  private readonly storageService: StorageService;

  /**
   * @section constructor
   */

  public constructor(
    storageService: StorageService,
    jobService: JobService,
    httpServerService: HttpServerService,
    pythonRuntimeService: PythonRuntimeService,
    jobPollIntervalMs: number,
  ) {
    this.storageService = storageService;
    this.jobService = jobService;
    this.httpServerService = httpServerService;
    this.pythonRuntimeService = pythonRuntimeService;
    this.jobPollIntervalMs = jobPollIntervalMs;
    this.intervalReference = null;
  }

  /**
   * @section factory
   */

  public static createDefault(): ServiceRuntime {
    const storageService = StorageService.createDefault();
    const modelService = ModelService.create(storageService);
    const pythonRuntimeService = PythonRuntimeService.createDefault();
    const jobService = JobService.create(storageService, modelService, pythonRuntimeService);
    const dashboardStateService = DashboardStateService.create(storageService);
    const appInfoService = AppInfoService.createDefault();
    const httpServerService = HttpServerService.create(appInfoService, modelService, jobService, dashboardStateService);
    storageService.initialize();
    jobService.recoverInterruptedJobs();
    return new ServiceRuntime(storageService, jobService, httpServerService, pythonRuntimeService, config.JOB_POLL_INTERVAL_MS);
  }

  /**
   * @section private:methods
   */

  private startJobLoop(): void {
    if (!this.intervalReference) {
      this.intervalReference = setInterval(() => {
        void this.jobService.processNextQueuedJob();
      }, this.jobPollIntervalMs);
    }
  }

  /**
   * @section public:methods
   */

  public buildServer(): ServerType {
    const server = this.httpServerService.buildServer();
    return server;
  }

  public startServer(): ServerType {
    this.pythonRuntimeService.verifyRuntime();
    const server = this.buildServer();
    this.startJobLoop();
    server.listen(config.DEFAULT_PORT, () => {
      logger.info(`service listening on http://localhost:${config.DEFAULT_PORT}`);
    });
    return server;
  }

  public dispose(): void {
    if (this.intervalReference) {
      clearInterval(this.intervalReference);
      this.intervalReference = null;
    }

    this.storageService.close();
  }
}
