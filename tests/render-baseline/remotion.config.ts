import { Config } from '@remotion/cli/config';
import path from 'path';

Config.setVideoImageFormat('png');
Config.setOverwriteOutput(true);

// Shared lib/ imports must resolve remotion/react from this project's node_modules
Config.overrideWebpackConfig((config) => ({
  ...config,
  resolve: {
    ...config.resolve,
    modules: [
      path.resolve(__dirname, 'node_modules'),
      ...(config.resolve?.modules || ['node_modules']),
    ],
  },
}));
