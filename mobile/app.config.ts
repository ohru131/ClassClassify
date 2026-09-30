import type { ExpoConfig } from 'expo/config'

// ストアに出すアプリの識別子。**公開後は変えないこと**（別アプリ扱いになり、購入の復元もできなくなる）。
const APP_ID = 'com.ohru131.mosaic'

/**
 * AdMob の App ID。未設定のときは Google がドキュメントで公開しているテスト用 App ID を使う。
 * EAS の production ビルドだけは未設定だと失敗させる（テスト ID のまま出荷する事故を防ぐため）。
 */
function admobPluginOptions() {
  const androidAppId = process.env.EXPO_PUBLIC_ADMOB_ANDROID_APP_ID
  const iosAppId = process.env.EXPO_PUBLIC_ADMOB_IOS_APP_ID
  const buildPlatform = process.env.EAS_BUILD_PLATFORM
  const missing = !androidAppId || (buildPlatform === 'ios' && !iosAppId)
  if (process.env.EAS_BUILD_PROFILE === 'production' && missing) {
    throw new Error('本番ビルドには AdMob の App ID（EXPO_PUBLIC_ADMOB_ANDROID_APP_ID / EXPO_PUBLIC_ADMOB_IOS_APP_ID）の設定が必須です。')
  }
  return {
    androidAppId: androidAppId || 'ca-app-pub-3940256099942544~3347511713',
    iosAppId: iosAppId || 'ca-app-pub-3940256099942544~1458002511',
    userTrackingUsageDescription: '無料版で表示する広告の配信と効果測定に使用します。',
  }
}

const config: ExpoConfig = {
  name: 'Mosaic クラス編成',
  slug: 'mosaic-class-classify',
  version: '1.0.0',
  // 学校のタブレット・Chromebook では横向き・分割画面で使うので、向きを固定しない。
  orientation: 'default',
  icon: './assets/icon.png',
  scheme: 'mosaic',
  userInterfaceStyle: 'light',
  ios: {
    supportsTablet: true,
    bundleIdentifier: APP_ID,
    infoPlist: { ITSAppUsesNonExemptEncryption: false },
  },
  android: {
    package: APP_ID,
    adaptiveIcon: { foregroundImage: './assets/adaptive-icon.png', backgroundColor: '#6366F1' },
    // 追加の権限は宣言しない（通知・カメラ・マイク等は使わない）。
    // ネイティブ依存（AdMob・課金）が持つ権限はマニフェストのマージで自動的に入る。
    permissions: [],
    // expo-file-system のプラグインが足す旧来のストレージ権限は使わない（取り込みは OS の
    // ファイル選択、書き出しはアプリのキャッシュ＋共有シートで、どちらも権限が要らない）。
    blockedPermissions: ['android.permission.READ_EXTERNAL_STORAGE', 'android.permission.WRITE_EXTERNAL_STORAGE'],
  },
  web: {
    bundler: 'metro',
    output: 'single',
    favicon: './assets/favicon.png',
  },
  plugins: [
    './plugins/withAdMobKotlinCompatibility',
    // タッチパネル必須にしない・リサイズ可能なウィンドウを妨げない（Chromebook・タブレット）
    './plugins/withLargeScreenSupport',
    // release ビルドから開発メニュー用の SYSTEM_ALERT_WINDOW を外す
    './plugins/withoutReleaseOverlayPermission',
    'expo-router',
    'expo-sharing',
    'expo-document-picker',
    ['expo-build-properties', { android: { minSdkVersion: 24 } }],
    ['react-native-google-mobile-ads', admobPluginOptions()],
  ],
  experiments: {
    typedRoutes: false,
  },
}

export default config
