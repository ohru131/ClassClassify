import type { ExpoConfig } from 'expo/config'

// ストアに出すアプリの識別子。**公開後は変えないこと**（別アプリ扱いになり、購入の復元もできなくなる）。
// 製品名は FairClass に改名したが、Play のアプリは改名前のこの ID で登録済みで変えられない。
const APP_ID = 'com.ohru131.mosaic'

const config: ExpoConfig = {
  // 端末の言語に合わせてホーム画面の名前を変える（locales/*.json）。既定は短い製品名
  name: 'FairClass',
  // EAS のプロジェクトと一致している必要があるので旧名のまま（利用者には見えない）
  slug: 'mosaic-class-classify',
  version: '1.6.0',
  // 学校のタブレット・Chromebook では横向き・分割画面で使うので、向きを固定しない。
  orientation: 'default',
  icon: './assets/icon.png',
  scheme: 'fairclass',
  userInterfaceStyle: 'light',
  ios: {
    supportsTablet: true,
    bundleIdentifier: APP_ID,
    infoPlist: { ITSAppUsesNonExemptEncryption: false },
  },
  android: {
    package: APP_ID,
    // **ローカルの gradle ビルド専用の versionCode。**
    // plugins/withLocalReleaseSigning.js が release を本番鍵で署名するので、ローカルでも
    // Play に出せる AAB が作れる。Play は同じ versionCode の AAB を二度受け付けないので、
    // リリースのたびに Play で未使用の値へ上げる（EAS の autoIncrement とは別系統）。
    versionCode: 8,
    adaptiveIcon: { foregroundImage: './assets/adaptive-icon.png', backgroundColor: '#6366F1' },
    // 追加の権限は宣言しない（通知・カメラ・マイク等は使わない）。
    // ネイティブ依存（課金 = expo-iap の Play Billing）が持つ BILLING 権限はマニフェストのマージで自動的に入る。
    permissions: [],
    // 名簿は端末の外へ出さない。Android の自動バックアップ（Google ドライブへのアプリデータの
    // バックアップ）に AsyncStorage の名簿が含まれないよう、アプリのバックアップを無効にする。
    allowBackup: false,
    // expo-file-system のプラグインが足す旧来のストレージ権限は使わない（取り込みは OS の
    // ファイル選択、書き出しはアプリのキャッシュ＋共有シートで、どちらも権限が要らない）。
    blockedPermissions: ['android.permission.READ_EXTERNAL_STORAGE', 'android.permission.WRITE_EXTERNAL_STORAGE'],
  },
  locales: {
    ja: './locales/ja.json',
    en: './locales/en.json',
    ko: './locales/ko.json',
    es: './locales/es.json',
    de: './locales/de.json',
    'pt-BR': './locales/pt-BR.json',
  },
  web: {
    bundler: 'metro',
    output: 'single',
    favicon: './assets/favicon.png',
  },
  plugins: [
    // タッチパネル必須にしない・リサイズ可能なウィンドウを妨げない（Chromebook・タブレット）
    './plugins/withLargeScreenSupport',
    // release ビルドから開発メニュー用の SYSTEM_ALERT_WINDOW を外す
    './plugins/withoutReleaseOverlayPermission',
    // debug を applicationId + ".debug" の別アプリにして Play 版と共存させる
    './plugins/withDebugPackageSuffix',
    // credentials.json があれば release を upload key で署名する（無ければ何もしない）
    './plugins/withLocalReleaseSigning',
    // release の R8 で最適化まで行う（既定の proguard-android.txt は -dontoptimize）
    './plugins/withR8Optimize',
    'expo-router',
    'expo-localization',
    'expo-sharing',
    'expo-document-picker',
    // 課金（Google Play Billing を直接使う。買い切り1本）
    'expo-iap',
    [
      'expo-build-properties',
      {
        android: {
          minSdkVersion: 24,
          // release を R8 で縮小・難読化する。無効だと Play Console の「DEX コードの最適化」が
          // しきい値（25%）を下回り、露出に影響すると警告される（期限 2027年2月）。
          enableMinifyInReleaseBuilds: true,
          enableShrinkResourcesInReleaseBuilds: true,
        },
      },
    ],
  ],
  experiments: {
    typedRoutes: false,
  },
  extra: {
    // EAS のプロジェクト（@ohru131/mosaic-class-classify）。upload key もここに保管している
    eas: {
      projectId: 'd7abe157-8b1d-45fe-8f97-802641b8d151',
    },
  },
}

export default config
