/**
 * @section imports:externals
 */

import { spawn } from "node:child_process";

/**
 * @section imports:internals
 */

import config from "../config.ts";
import type { PythonExecutionResult, PythonJobCommand, PythonJobExecutor } from "./index.ts";

/**
 * @section class
 */

export class PythonRuntimeService {
  /**
   * @section private:attributes
   */

  private readonly executor: PythonJobExecutor;

  private readonly pythonBin: string;

  private readonly workerScriptPath: string;

  /**
   * @section constructor
   */

  public constructor(pythonBin: string, workerScriptPath: string, executor?: PythonJobExecutor) {
    this.pythonBin = pythonBin;
    this.workerScriptPath = workerScriptPath;
    this.executor = executor || this.executeThroughPython.bind(this);
  }

  /**
   * @section factory
   */

  public static createDefault(): PythonRuntimeService {
    const service = new PythonRuntimeService(config.PYTHON_BIN, config.PYTHON_WORKER_SCRIPT);
    return service;
  }

  /**
   * @section private:methods
   */

  private async executeThroughPython(command: PythonJobCommand): Promise<PythonExecutionResult> {
    const result = await new Promise<PythonExecutionResult>((resolve) => {
      let errorOutput = "";
      const process = spawn(this.pythonBin, [this.workerScriptPath, command.action, command.requestPath, command.resultPath]);
      process.stderr.on("data", (chunk) => {
        errorOutput += String(chunk);
      });
      process.on("close", (exitCode) => {
        const normalizedErrorOutput = errorOutput.trim();
        const executionResult =
          exitCode === 0
            ? { errorMessage: null, isSuccess: true }
            : {
                errorMessage: normalizedErrorOutput || "python worker execution failed",
                isSuccess: false,
              };
        resolve(executionResult);
      });
      process.on("error", (error) => {
        resolve({ errorMessage: error.message, isSuccess: false });
      });
    });
    return result;
  }

  /**
   * @section public:methods
   */

  public async execute(command: PythonJobCommand): Promise<PythonExecutionResult> {
    const result = await this.executor(command);
    return result;
  }
}
