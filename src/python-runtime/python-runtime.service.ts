/**
 * @section imports:externals
 */

import { spawn, spawnSync } from "node:child_process";

/**
 * @section imports:internals
 */

import config from "../config.ts";
import logger from "../logger.ts";
import type {
  PythonExecutionResult,
  PythonJobCommand,
  PythonJobExecutor,
  PythonPredictionCommand,
  PythonPredictionExecutor,
  PythonPredictionResult,
} from "./index.ts";

/**
 * @section class
 */

export class PythonRuntimeService {
  /**
   * @section private:attributes
   */

  private readonly executor: PythonJobExecutor;

  private readonly predictionExecutor: PythonPredictionExecutor;

  private readonly pythonBin: string;

  private readonly workerScriptPath: string;

  /**
   * @section constructor
   */

  public constructor(pythonBin: string, workerScriptPath: string, executor?: PythonJobExecutor, predictionExecutor?: PythonPredictionExecutor) {
    this.pythonBin = pythonBin;
    this.workerScriptPath = workerScriptPath;
    this.executor = executor || this.executeThroughPython.bind(this);
    this.predictionExecutor = predictionExecutor || this.executePredictionThroughPython.bind(this);
  }

  /**
   * @section static:properties
   */

  private static readonly TENSORFLOW_IMPORT_CHECK = "import tensorflow as tf; print(tf.__version__)";

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

  private async executePredictionThroughPython(command: PythonPredictionCommand): Promise<PythonPredictionResult> {
    const result = await new Promise<PythonPredictionResult>((resolve) => {
      let errorOutput = "";
      let standardOutput = "";
      const process = spawn(this.pythonBin, [this.workerScriptPath, "predict-model-stdio"]);
      process.stdout.on("data", (chunk) => {
        standardOutput += String(chunk);
      });
      process.stderr.on("data", (chunk) => {
        errorOutput += String(chunk);
      });
      process.on("close", (exitCode) => {
        const normalizedErrorOutput = errorOutput.trim();
        const normalizedStandardOutput = standardOutput.trim();
        let executionResult: PythonPredictionResult;

        if (exitCode !== 0) {
          executionResult = {
            errorMessage: normalizedErrorOutput || "python worker execution failed",
            isSuccess: false,
          };
        } else {
          try {
            executionResult = {
              isSuccess: true,
              result: JSON.parse(normalizedStandardOutput) as Record<string, unknown>,
            };
          } catch (error) {
            const normalizedError = error instanceof Error ? error.message : "python worker returned an invalid prediction payload";
            logger.error(`python worker returned an invalid prediction payload: ${normalizedError}`);
            executionResult = {
              errorMessage: normalizedError,
              isSuccess: false,
            };
          }
        }

        resolve(executionResult);
      });
      process.on("error", (error) => {
        resolve({ errorMessage: error.message, isSuccess: false });
      });
      process.stdin.write(JSON.stringify(command));
      process.stdin.end();
    });
    return result;
  }

  private buildRuntimeCheckErrorMessage(errorText: string): string {
    const normalizedErrorText = errorText.trim();
    const message = normalizedErrorText || `tensorflow is not available in python runtime '${this.pythonBin}'`;
    return message;
  }

  /**
   * @section public:methods
   */

  public verifyRuntime(): void {
    const commandResult = spawnSync(this.pythonBin, ["-c", PythonRuntimeService.TENSORFLOW_IMPORT_CHECK], {
      encoding: "utf8",
    });
    const standardError = typeof commandResult.stderr === "string" ? commandResult.stderr : "";
    const processError = commandResult.error ? commandResult.error.message : "";
    const errorText = `${standardError}\n${processError}`.trim();

    if (commandResult.status !== 0 || commandResult.error) {
      throw new Error(`python runtime check failed for '${this.pythonBin}': ${this.buildRuntimeCheckErrorMessage(errorText)}`);
    }
  }

  public async execute(command: PythonJobCommand): Promise<PythonExecutionResult> {
    const result = await this.executor(command);
    return result;
  }

  public async executePrediction(command: PythonPredictionCommand): Promise<PythonPredictionResult> {
    const result = await this.predictionExecutor(command);
    return result;
  }
}
