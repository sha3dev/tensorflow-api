module.exports = {
  apps: [
    {
      name: "@sha3/tensorflow-api",
      script: "npm",
      args: "run start",
      interpreter: "none",
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
