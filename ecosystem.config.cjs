module.exports = { apps: [{ name: "@sha3/tensorflow-api", script: "node", args: "--import tsx src/main.ts", env: { NODE_ENV: "production" } }] };
