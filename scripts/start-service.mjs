import { spawn } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const ROOT_DIR = path.resolve(path.dirname(SCRIPT_PATH), "..");
const VENV_DIR = path.join(ROOT_DIR, ".venv");
const REQUIREMENTS_PATH = path.join(ROOT_DIR, "requirements", "python-runtime.txt");
const DEFAULT_PYTHON_VERSION = process.env.TENSORFLOW_API_PYTHON_VERSION || "3.11";
const UV_INSTALL_DIR = process.env.TENSORFLOW_API_UV_INSTALL_DIR || path.join(homedir(), ".cache", "tensorflow-api", "uv");
const PYTHON_BUILD_TOOLS = ["pip", "setuptools", "wheel"];
const TENSORFLOW_IMPORT_CHECK = "import tensorflow as tf; print(tf.__version__)";

function getUvExecutablePath() {
  const executablePath = process.platform === "win32" ? path.join(UV_INSTALL_DIR, "uv.exe") : path.join(UV_INSTALL_DIR, "uv");
  return executablePath;
}

function getVenvPythonPath() {
  const pythonPath = process.platform === "win32" ? path.join(VENV_DIR, "Scripts", "python.exe") : path.join(VENV_DIR, "bin", "python");
  return pythonPath;
}

function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT_DIR,
      env: { ...process.env, ...(options.env || {}) },
      shell: options.shell || false,
      stdio: options.stdio || "inherit",
    });

    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command} exited with code ${code ?? 1}`));
      }
    });
  });
}

function runShellCommand(command, env = {}) {
  const shell = process.platform === "win32" ? "powershell.exe" : "sh";
  const args = process.platform === "win32" ? ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command] : ["-c", command];
  return runCommand(shell, args, { env, shell: false });
}

async function canRun(command, args) {
  let canExecute = true;

  try {
    await runCommand(command, args, { stdio: "ignore" });
  } catch {
    canExecute = false;
  }

  return canExecute;
}

async function ensureUv() {
  const configuredUvPath = process.env.UV_BIN;
  const cachedUvPath = getUvExecutablePath();
  let uvPath = configuredUvPath && existsSync(configuredUvPath) ? configuredUvPath : null;

  if (!uvPath && existsSync(cachedUvPath)) {
    uvPath = cachedUvPath;
  }

  if (!uvPath) {
    const canUseSystemUv = await canRun("uv", ["--version"]);

    if (canUseSystemUv) {
      uvPath = "uv";
    }
  }

  if (!uvPath) {
    const installEnv = { UV_UNMANAGED_INSTALL: UV_INSTALL_DIR };

    if (process.platform === "win32") {
      await runShellCommand("irm https://astral.sh/uv/install.ps1 | iex", installEnv);
    } else {
      await runShellCommand("curl -LsSf https://astral.sh/uv/install.sh | sh", installEnv);
    }

    uvPath = getUvExecutablePath();
  }

  if (!uvPath || (!existsSync(uvPath) && uvPath !== "uv")) {
    throw new Error("failed to install or locate uv");
  }

  return uvPath;
}

async function bootstrapPythonRuntime(uvPath) {
  let pythonPath = getVenvPythonPath();
  let hasRetriedBootstrap = false;

  while (true) {
    if (!existsSync(pythonPath)) {
      await runCommand(uvPath, ["venv", VENV_DIR, "--python", DEFAULT_PYTHON_VERSION]);
    }

    try {
      await runCommand(uvPath, ["pip", "install", "--python", pythonPath, "--upgrade", ...PYTHON_BUILD_TOOLS]);
      await runCommand(uvPath, ["pip", "install", "--python", pythonPath, "-r", REQUIREMENTS_PATH]);
      await runCommand(pythonPath, ["-c", TENSORFLOW_IMPORT_CHECK]);
      break;
    } catch (bootstrapError) {
      if (hasRetriedBootstrap) {
        throw bootstrapError;
      }

      console.warn("Python runtime bootstrap failed. Recreating .venv and retrying once.");
      rmSync(VENV_DIR, { force: true, recursive: true });
      pythonPath = getVenvPythonPath();
      hasRetriedBootstrap = true;
    }
  }

  return pythonPath;
}

async function startService(pythonPath) {
  await runCommand(process.execPath, ["--import", "tsx", "src/main.ts"], {
    env: {
      PYTHON_BIN: pythonPath,
    },
  });
}

async function main() {
  const uvPath = await ensureUv();
  const pythonPath = await bootstrapPythonRuntime(uvPath);
  await startService(pythonPath);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
