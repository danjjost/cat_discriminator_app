module.exports = function (api) {
  api.cache(true);
  return {
    presets: ["babel-preset-expo"],
    plugins: [
      [
        "module:react-native-dotenv",
        {
          moduleName: "@env",
          path: ".env",
          blacklist: null, // or blocklist
          whitelist: null, // or allowlist
          safe: false,
          allowUndefined: true,
          verbose: false,
        },
      ],
    ],
  };
};
