const { withAndroidManifest } = require("expo/config-plugins");

// 学校の Chromebook（Play ストアの Android アプリとして動く）とタブレットで使えるようにする。
//
// Android はマニフェストに何も書かないと android.hardware.touchscreen を「必須」とみなし、
// Play はタッチパネルの無い Chromebook にアプリを配信しない。操作はすべてクリック・キーボードで
// 完結する（長押し・ドラッグ・スワイプ前提の操作は無い）ので、必須ではないと明示する。
//
// 画面の向きは app.config.ts の orientation: "default" で固定しない。resizeableActivity は
// 既定（true）のままにして、フリーフォームのウィンドウと分割画面を妨げない。
const FEATURES = ["android.hardware.touchscreen"];

module.exports = function withLargeScreenSupport(config) {
  return withAndroidManifest(config, (config) => {
    const manifest = config.modResults.manifest;
    const features = (manifest["uses-feature"] ??= []);
    for (const name of FEATURES) {
      const existing = features.find((f) => f.$?.["android:name"] === name);
      if (existing) existing.$["android:required"] = "false";
      else features.push({ $: { "android:name": name, "android:required": "false" } });
    }
    const app = manifest.application?.[0];
    if (app?.$?.["android:resizeableActivity"] === "false") delete app.$["android:resizeableActivity"];
    for (const activity of app?.activity ?? []) {
      if (activity.$?.["android:resizeableActivity"] === "false") delete activity.$["android:resizeableActivity"];
    }
    return config;
  });
};
