#!/usr/bin/env node
// ローカルで作った署名済み AAB を Google Play のトラックへ上げる。
//
//   node scripts/upload-play-aab.mjs                          # ドライラン（既定。通信しない）
//   node scripts/upload-play-aab.mjs --commit --key sa.json   # アップロードして内部テストのドラフトを作る
//   オプション: --aab <path> / --track internal|alpha|beta|production / --status draft|completed / --package <id> / --name <リリース名>
//
// 既定は internal・draft。**公開前のアプリ（ストアの設定が未完了）は draft でしか作れない**ので、
// 公開できる状態になったら Play Console でドラフトを確認して「リリースを開始」する。
// 一度送った versionCode は edit を捨てても使用済みとして残りうるので、「お試しで送る」モードは無い。
import { createHash, createSign } from 'node:crypto'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const API = 'https://androidpublisher.googleapis.com/androidpublisher/v3'
const UPLOAD_API = 'https://androidpublisher.googleapis.com/upload/androidpublisher/v3'
// mobile/app.config.ts の APP_ID と同じ値
const DEFAULT_PACKAGE = 'com.ohru131.mosaic'
const DEFAULT_AAB = join(ROOT, 'mobile', 'android', 'app', 'build', 'outputs', 'bundle', 'release', 'app-release.aab')
const DEFAULT_KEY = join(ROOT, 'play-service-account.json')
const TRACKS = ['internal', 'alpha', 'beta', 'production']

function parseArgs(argv) {
  const args = { commit: false, aab: DEFAULT_AAB, track: 'internal', status: 'draft', package: DEFAULT_PACKAGE, key: null, name: null }
  const values = { '--aab': 'aab', '--track': 'track', '--status': 'status', '--package': 'package', '--key': 'key', '--name': 'name' }
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i]
    if (flag in values) {
      const next = argv[i + 1]
      if (next === undefined || next.startsWith('-')) throw new Error(`${flag} に値がない`)
      args[values[flag]] = next
      i += 1
    } else if (flag === '--commit') args.commit = true
    else throw new Error(`不明な引数: ${flag}`)
  }
  if (!TRACKS.includes(args.track)) throw new Error(`track は ${TRACKS.join(' / ')} のいずれか`)
  if (!['draft', 'completed'].includes(args.status)) throw new Error('status は draft か completed')
  return args
}

// サービスアカウントの JSON 鍵から RS256 の JWT を作ってアクセストークンへ交換する
async function getAccessToken(keyPath) {
  const key = JSON.parse(readFileSync(keyPath, 'utf8'))
  const now = Math.floor(Date.now() / 1000)
  const b64 = (o) => Buffer.from(JSON.stringify(o)).toString('base64url')
  const aud = key.token_uri ?? 'https://oauth2.googleapis.com/token'
  const unsigned = `${b64({ alg: 'RS256', typ: 'JWT' })}.${b64({ iss: key.client_email, scope: 'https://www.googleapis.com/auth/androidpublisher', aud, iat: now, exp: now + 3600 })}`
  const signer = createSign('RSA-SHA256')
  signer.update(unsigned)
  const res = await fetch(aud, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer', assertion: `${unsigned}.${signer.sign(key.private_key, 'base64url')}` }),
  })
  const body = await res.json()
  if (!res.ok) throw new Error(`トークン取得に失敗 (${res.status}): ${JSON.stringify(body)}`)
  return body.access_token
}

async function api(token, method, url, { json, body, contentType } = {}) {
  const res = await fetch(url.startsWith('http') ? url : `${API}${url}`, {
    method,
    headers: { authorization: `Bearer ${token}`, ...(json ? { 'content-type': 'application/json' } : {}), ...(contentType ? { 'content-type': contentType } : {}) },
    body: json ? JSON.stringify(json) : body,
  })
  const text = await res.text()
  let parsed = null
  try { parsed = text ? JSON.parse(text) : null } catch { /* 空応答 */ }
  if (!res.ok) throw new Error(`${method} ${url} が ${res.status}: ${parsed?.error?.message ?? text.slice(0, 400)}`)
  return parsed
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (!existsSync(args.aab)) throw new Error(`AAB が無い: ${args.aab}（cd mobile/android && ./gradlew.bat bundleRelease）`)
  const buf = readFileSync(args.aab)
  const sha1 = createHash('sha1').update(buf).digest('hex')
  const version = JSON.parse(readFileSync(join(ROOT, 'mobile', 'package.json'), 'utf8')).version
  const name = args.name ?? version

  console.log(`モード   ${args.commit ? '★commit（アップロードする）' : 'dry-run（通信しない）'}`)
  console.log(`package  ${args.package}`)
  console.log(`AAB      ${args.aab}（${(statSync(args.aab).size / 1048576).toFixed(1)}MB, sha1 ${sha1}）`)
  console.log(`リリース ${name} → ${args.track} / ${args.status}`)
  if (!args.commit) {
    console.log('\nドライラン。送るなら --commit を付ける。')
    return
  }

  const keyPath = args.key ?? process.env.GOOGLE_PLAY_SERVICE_ACCOUNT_JSON ?? DEFAULT_KEY
  if (!existsSync(keyPath)) throw new Error(`鍵が見つからない: ${keyPath}`)
  const token = await getAccessToken(keyPath)
  const edit = await api(token, 'POST', `/applications/${args.package}/edits`)
  console.log(`\nedit ${edit.id} を作った`)
  try {
    const uploaded = await api(token, 'POST', `${UPLOAD_API}/applications/${args.package}/edits/${edit.id}/bundles?uploadType=media`, {
      body: buf,
      contentType: 'application/octet-stream',
    })
    console.log(`送った: versionCode ${uploaded.versionCode} sha1 ${uploaded.sha1}`)
    if (uploaded.sha1 !== sha1) throw new Error('Play が返した sha1 がローカルと違う')
    await api(token, 'PUT', `/applications/${args.package}/edits/${edit.id}/tracks/${args.track}`, {
      json: { track: args.track, releases: [{ name, versionCodes: [String(uploaded.versionCode)], status: args.status }] },
    })
    await api(token, 'POST', `/applications/${args.package}/edits/${edit.id}:commit`)
    console.log(`\ncommit した。Play Console の「テストとリリース」→ ${args.track} に ${args.status} のリリースができた。`)
  } catch (e) {
    await api(token, 'DELETE', `/applications/${args.package}/edits/${edit.id}`).catch(() => {})
    throw e
  }
}

main().catch((e) => {
  console.error(`\n失敗: ${e.message}`)
  process.exitCode = 1
})
