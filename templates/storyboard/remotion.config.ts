import { Config } from '@remotion/cli/config';
import path from 'path';

Config.setOverwriteOutput(true);
Config.overrideWebpackConfig((config) => ({
  ...config,
  resolve: {
    ...config.resolve,
    modules: [path.resolve(__dirname, 'node_modules'), ...(config.resolve?.modules || ['node_modules'])],
  },
}));
