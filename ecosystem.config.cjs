module.exports = {
  apps: [
    {
      name: "@sha3/tensorflow-api",
      script: "npm",
      args: "run start",
      interpreter: "none",
      env: {
        CUDA_VISIBLE_DEVICES: "",
        NODE_ENV: "production",
        TF_CPP_MIN_LOG_LEVEL: "2",
        TF_ENABLE_ONEDNN_OPTS: "0",
      },
    },
  ],
};
