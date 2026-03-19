import "dotenv/config";

const ENV = process.env;

const config = {
  RESPONSE_CONTENT_TYPE: ENV.RESPONSE_CONTENT_TYPE || "application/json",
  DEFAULT_PORT: Number(ENV.PORT || 3100),
  SERVICE_NAME: ENV.SERVICE_NAME || "@sha3/tensorflow-api",
  STORAGE_ROOT: ENV.STORAGE_ROOT || "var/storage",
  SQLITE_PATH: ENV.SQLITE_PATH || "var/storage/tensorflow-api.sqlite",
  PYTHON_BIN: ENV.PYTHON_BIN || "python3",
  PYTHON_WORKER_SCRIPT: ENV.PYTHON_WORKER_SCRIPT || "python/tensorflow_api_worker.py",
  JOB_POLL_INTERVAL_MS: Number(ENV.JOB_POLL_INTERVAL_MS || 1000),
} as const;

export default config;
