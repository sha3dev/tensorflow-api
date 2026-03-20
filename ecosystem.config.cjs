module.exports = {
  apps: [
    {
      name: "@sha3/tensorflow-api",
      script: "npm",
      args: "run start",
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
