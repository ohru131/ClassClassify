// スマホ版は Web 版と同じソルバー（../src/solver）を共有する。
// - watchFolders: リポジトリ直下の src/solver をバンドル対象に入れる
// - resolveRequest: 共有コードが import する xlsx-js-style を、リポジトリ直下ではなく
//   mobile/node_modules から解決する（EAS ビルドでは直下の node_modules が無いため）
// - xlsx-js-style が require する旧 .xls 用の文字コード表（約470KB）はスタブに差し替える
//   （Web 版の vite.config.ts と同じ扱い。.xlsx は UTF-8 のみなので不要）
const { getDefaultConfig } = require('expo/metro-config')
const path = require('path')

const projectRoot = __dirname
const repoRoot = path.resolve(projectRoot, '..')
const sharedSolver = path.join(repoRoot, 'src', 'solver')
const cpexcelStub = path.join(projectRoot, 'stubs', 'cpexcel.cjs')
const SHARED_DEPENDENCIES = new Set(['xlsx-js-style'])

const config = getDefaultConfig(projectRoot)
config.watchFolders = [...(config.watchFolders ?? []), sharedSolver]
config.resolver.nodeModulesPaths = [path.join(projectRoot, 'node_modules')]

const upstream = config.resolver.resolveRequest
config.resolver.resolveRequest = (context, moduleName, platform) => {
  const resolve = upstream ?? context.resolveRequest
  if (moduleName === './cpexcel.js' && context.originModulePath.includes(`${path.sep}xlsx-js-style${path.sep}`)) {
    return { type: 'sourceFile', filePath: cpexcelStub }
  }
  if (SHARED_DEPENDENCIES.has(moduleName)) {
    return resolve({ ...context, originModulePath: path.join(projectRoot, 'package.json') }, moduleName, platform)
  }
  return resolve(context, moduleName, platform)
}

module.exports = config
