/**
 * @section imports:internals
 */

import type { StorageService } from "../storage/index.ts";
import type { DashboardStatePayload } from "./index.ts";

/**
 * @section class
 */

export class DashboardStateService {
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

  public static create(storageService: StorageService): DashboardStateService {
    const service = new DashboardStateService(storageService);
    return service;
  }

  /**
   * @section public:methods
   */

  public buildState(): DashboardStatePayload {
    const models = this.storageService.listModelRecords();
    const recentJobs = this.storageService.listJobRecords({ limit: 20 });
    const statusSummary = this.storageService.countJobStatuses();
    const payload: DashboardStatePayload = {
      models,
      recentJobs,
      summary: {
        failedJobCount: statusSummary.failedJobCount,
        modelCount: models.length,
        queuedJobCount: statusSummary.queuedJobCount,
        runningJobCount: statusSummary.runningJobCount,
      },
    };
    return payload;
  }
}
