const { withAppBuildGradle } = require("expo/config-plugins");

// release の R8 で最適化まで行う（既定の proguard-android.txt → proguard-android-optimize.txt）。
//
// prebuild が生成する app/build.gradle は `getDefaultProguardFile("proguard-android.txt")` を使う。
// **この既定ファイルには `-dontoptimize` が入っている**ので、expo-build-properties で
// enableMinifyInReleaseBuilds を立てても R8 は縮小と難読化だけで、最適化の工程を飛ばす。
// Play Console の「DEX コードの最適化」は最適化も測るので、Google の推奨どおり
// -optimize 版に替える（AGP 9 では proguard-android.txt 自体が使えなくなる）。
//
// 最適化はインライン化・クラスの統合をするので、リフレクションで呼ばれるクラスの keep 漏れが
// 表に出やすくなる。ネイティブの依存を足したり上げたりしたら release を実機で確かめる。
//
// **android/ は .gitignore 済みの生成物**なので、build.gradle を直に書き換えても次の prebuild で戻る。

const from = 'getDefaultProguardFile("proguard-android.txt")';
const to = 'getDefaultProguardFile("proguard-android-optimize.txt")';

module.exports = function withR8Optimize(config) {
  return withAppBuildGradle(config, (config) => {
    if (config.modResults.language !== "groovy") {
      throw new Error("withR8Optimize requires a Groovy app/build.gradle");
    }
    const contents = config.modResults.contents;
    if (contents.includes(to)) return config;
    // テンプレートが変わって置き換え先が見つからないときは黙って通さない（最適化なしに戻るため）
    if (!contents.includes(from)) {
      throw new Error(`withR8Optimize could not find ${from} in app/build.gradle`);
    }
    config.modResults.contents = contents.replace(from, to);
    return config;
  });
};
