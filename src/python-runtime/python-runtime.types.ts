export type PythonJobCommand = {
  action: "create-model" | "predict-model" | "train-model";
  requestPath: string;
  resultPath: string;
};

export type PythonExecutionResult = {
  errorMessage: string | null;
  isSuccess: boolean;
};

export type PythonJobExecutor = (command: PythonJobCommand) => Promise<PythonExecutionResult>;
