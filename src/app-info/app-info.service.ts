/**
 * @section imports:internals
 */

import config from "../config.ts";

/**
 * @section types
 */

export type AppInfoPayload = {
  ok: true;
  serviceName: string;
  dashboardPath: string;
  statePath: string;
};

/**
 * @section class
 */

export class AppInfoService {
  /**
   * @section private:attributes
   */

  private readonly serviceName: string;

  /**
   * @section constructor
   */

  public constructor(serviceName: string) {
    this.serviceName = serviceName;
  }

  /**
   * @section factory
   */

  public static createDefault(): AppInfoService {
    return new AppInfoService(config.SERVICE_NAME);
  }

  /**
   * @section public:methods
   */

  public buildPayload(): AppInfoPayload {
    return { ok: true, serviceName: this.serviceName, dashboardPath: "/dashboard", statePath: "/api/state" };
  }
}
