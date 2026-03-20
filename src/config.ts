import "dotenv/config";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ENV = process.env;
const CONFIG_FILE_PATH = fileURLToPath(import.meta.url);
const PROJECT_ROOT = resolve(dirname(CONFIG_FILE_PATH), "..");
const DEFAULT_VENV_PYTHON_PATH =
  process.platform === "win32" ? join(PROJECT_ROOT, ".venv", "Scripts", "python.exe") : join(PROJECT_ROOT, ".venv", "bin", "python");
const DEFAULT_PYTHON_BIN = existsSync(DEFAULT_VENV_PYTHON_PATH) ? DEFAULT_VENV_PYTHON_PATH : "python3";

const config = {
  RESPONSE_CONTENT_TYPE: ENV.RESPONSE_CONTENT_TYPE || "application/json",
  DEFAULT_PORT: Number(ENV.PORT || 3100),
  SERVICE_NAME: ENV.SERVICE_NAME || "@sha3/tensorflow-api",
  STORAGE_ROOT: ENV.STORAGE_ROOT || "var/storage",
  SQLITE_PATH: ENV.SQLITE_PATH || "var/storage/tensorflow-api.sqlite",
  PYTHON_BIN: ENV.PYTHON_BIN || DEFAULT_PYTHON_BIN,
  PYTHON_WORKER_SCRIPT: ENV.PYTHON_WORKER_SCRIPT || "python/tensorflow_api_worker.py",
  JOB_POLL_INTERVAL_MS: Number(ENV.JOB_POLL_INTERVAL_MS || 1000),
} as const;

export default config;
