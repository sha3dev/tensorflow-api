export type DashboardStatePayload = {
  models: import("../model/index.ts").ModelRecord[];
  recentJobs: import("../job/index.ts").JobRecord[];
  summary: {
    failedJobCount: number;
    modelCount: number;
    queuedJobCount: number;
    runningJobCount: number;
  };
};
