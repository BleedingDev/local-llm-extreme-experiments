import { chmodSync } from "node:fs";
import rspack, { type Configuration } from "@rspack/core";

const config: Configuration = {
  mode: "production",
  target: "node",
  entry: {
    index: "./src/index.ts",
  },
  output: {
    filename: "[name].js",
    path: new URL("dist", import.meta.url).pathname,
    module: true,
    clean: true,
    chunkFormat: "module",
  },
  experiments: {
    outputModule: true,
  },
  externalsPresets: {
    node: true,
  },
  externalsType: "module",
  externals: [
    ({ request }, callback) => {
      if (request === "@ax-llm/ax" || request === "zod") {
        callback(null, request);
        return;
      }
      if (request != null && request.startsWith("node:")) {
        callback(null, request);
        return;
      }
      callback();
    },
  ],
  resolve: {
    extensions: [".ts", ".tsx", ".js", ".mjs", ".json"],
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        loader: "builtin:swc-loader",
        options: {
          jsc: {
            parser: {
              syntax: "typescript",
            },
            target: "es2022",
          },
        },
      },
    ],
  },
  plugins: [
    new rspack.BannerPlugin({
      banner: "#!/usr/bin/env node",
      raw: true,
    }),
    {
      apply(compiler) {
        compiler.hooks.afterEmit.tap("BagExecutablePlugin", () => {
          chmodSync(new URL("dist/index.js", import.meta.url), 0o755);
        });
      },
    },
  ],
};

export default config;
