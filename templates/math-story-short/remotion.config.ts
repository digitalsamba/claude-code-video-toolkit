import { Config } from "@remotion/cli/config";
import path from "path";

Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);

// Ensure shared lib/ imports resolve packages from this project's node_modules
// Without this, imports like `@remotion/transitions` in lib/transitions/ fail
// because webpack resolves them relative to lib/, not the project root.
Config.overrideWebpackConfig((config) => {
  return {
    ...config,
    resolve: {
      ...config.resolve,
      modules: [
        path.resolve(__dirname, "node_modules"),
        ...(config.resolve?.modules || ["node_modules"]),
      ],
    },
  };
});
