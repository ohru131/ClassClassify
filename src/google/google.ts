/**
 * Google スプレッドシート連携（Google Identity Services + Google Picker + Drive/Sheets API）。
 * スコープは drive.file のみ: ユーザーが Picker で選んだファイルと、このアプリが作成したファイルにだけアクセスできる。
 * 通信はブラウザ ⇔ Google の間だけで行い、第三者のサーバーは経由しない。
 */

import { isAppLanguage, type AppLanguage } from '../i18n/languages'

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined
const API_KEY = import.meta.env.VITE_GOOGLE_API_KEY as string | undefined
/** Google Cloud プロジェクト番号。Picker で選んだファイルへのアクセス権付与に必要 */
const APP_ID = import.meta.env.VITE_GOOGLE_APP_ID as string | undefined

const SCOPE = 'https://www.googleapis.com/auth/drive.file'
const SHEET_MIME = 'application/vnd.google-apps.spreadsheet'
const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

export const googleEnabled = !!(CLIENT_ID && API_KEY && APP_ID)

/** エラー・Picker の文言（既定は日本語。App が選択中の言語に差し替える） */
interface GoogleMessages {
  scriptFailed: (src: string) => string
  authFailed: (msg: string) => string
  apiError: (msg: string) => string
  pickerTitle: string
  /** Picker の表示言語（Google の言語コード） */
  locale?: string
}
let MSG: GoogleMessages = {
  scriptFailed: (src) => `${src} を読み込めませんでした`,
  authFailed: (msg) => `Google 認証に失敗しました: ${msg}`,
  apiError: (msg) => `Google API エラー: ${msg}`,
  pickerTitle: '名簿のスプレッドシートを選択',
  locale: 'ja',
}
export function setGoogleMessages(m: GoogleMessages) {
  MSG = { ...MSG, ...m }
}

/* eslint-disable @typescript-eslint/no-explicit-any */
declare global {
  interface Window {
    google?: any
    gapi?: any
  }
}

export interface GoogleFile {
  id: string
  name: string
  mimeType: string
  url: string
}

const loaded = new Map<string, Promise<void>>()
function loadScript(src: string) {
  let p = loaded.get(src)
  if (!p) {
    p = new Promise<void>((resolve, reject) => {
      const s = document.createElement('script')
      s.src = src
      s.async = true
      s.onload = () => resolve()
      s.onerror = () => {
        loaded.delete(src)
        reject(new Error(MSG.scriptFailed(src)))
      }
      document.head.appendChild(s)
    })
    loaded.set(src, p)
  }
  return p
}

let token: { value: string; expires: number } | null = null
/** 一度同意を得たら、以降の再取得は同意画面を出さない（ポップアップブロック回避） */
let consented = false

async function getToken(): Promise<string> {
  if (token && token.expires > Date.now() + 60_000) return token.value
  await loadScript('https://accounts.google.com/gsi/client')
  return new Promise((resolve, reject) => {
    const client = window.google.accounts.oauth2.initTokenClient({
      client_id: CLIENT_ID,
      scope: SCOPE,
      callback: (res: any) => {
        if (res.error) return reject(new Error(MSG.authFailed(res.error_description ?? res.error)))
        token = { value: res.access_token, expires: Date.now() + Number(res.expires_in) * 1000 }
        consented = true
        resolve(res.access_token)
      },
      error_callback: (err: any) =>
        reject(new Error(err?.type === 'popup_closed' ? 'cancelled' : MSG.authFailed(err?.message ?? err?.type))),
    })
    client.requestAccessToken({ prompt: consented ? '' : undefined })
  })
}

async function api(url: string, init: RequestInit = {}, retried = false): Promise<Response> {
  const t = await getToken()
  const res = await fetch(url, { ...init, headers: { ...init.headers, Authorization: `Bearer ${t}` } })
  if (res.status === 401 && !retried) {
    // トークン失効 → 取り直して1回だけ再試行
    token = null
    return api(url, init, true)
  }
  if (!res.ok) {
    let msg = `${res.status}`
    try {
      msg = (await res.json()).error?.message ?? msg
    } catch {
      /* ignore */
    }
    throw new Error(MSG.apiError(msg))
  }
  return res
}

/** Picker でスプレッドシート（または Drive 上の .xlsx）を1つ選ぶ。キャンセル時は null */
export async function pickSpreadsheet(): Promise<GoogleFile | null> {
  const t = await getToken()
  await loadScript('https://apis.google.com/js/api.js')
  await new Promise<void>((resolve) => window.gapi.load('picker', () => resolve()))
  const g = window.google.picker
  return new Promise((resolve) => {
    const view = new g.DocsView(g.ViewId.SPREADSHEETS).setMimeTypes(`${SHEET_MIME},${XLSX_MIME}`).setMode(g.DocsViewMode.LIST)
    const picker = new g.PickerBuilder()
      .addView(view)
      .setOAuthToken(t)
      .setDeveloperKey(API_KEY)
      .setAppId(APP_ID)
      .setLocale(MSG.locale ?? 'ja')
      .setTitle(MSG.pickerTitle)
      .setCallback((data: any) => {
        if (data.action === g.Action.PICKED) {
          const d = data.docs[0]
          resolve({ id: d.id, name: d.name, mimeType: d.mimeType, url: d.url })
        } else if (data.action === g.Action.CANCEL) resolve(null)
      })
      .build()
    picker.setVisible(true)
  })
}

/** 選んだファイルを .xlsx のバイト列として取得（Google スプレッドシートは xlsx にエクスポート） */
export async function downloadAsXlsx(file: GoogleFile): Promise<ArrayBuffer> {
  const id = encodeURIComponent(file.id)
  const url =
    file.mimeType === SHEET_MIME
      ? `https://www.googleapis.com/drive/v3/files/${id}/export?mimeType=${encodeURIComponent(XLSX_MIME)}`
      : `https://www.googleapis.com/drive/v3/files/${id}?alt=media`
  return (await api(url)).arrayBuffer()
}

type Cell = string | number | null

export interface SheetSpec {
  name: string
  rows: Cell[][]
  frozenRows?: number
  frozenCols?: number
  /** 背景色を付ける行（0始まり）と色 */
  bands?: { row: number; color: string; bold?: boolean }[]
  /** 太字にする列（0始まり） */
  boldCols?: number[]
  /** 列幅（px） */
  colWidths?: number[]
  /** セル単位の塗り（結果出力用） */
  fills?: { row: number; col: number; cols?: number; bg?: string; fg?: string; bold?: boolean }[]
  /** 列幅（文字数。colWidths が無いときに使う） */
  widths?: number[]
}

/** SheetSpec の書式を Sheets API の batchUpdate リクエストにする */
function formatRequests(sheetId: number, t: SheetSpec): unknown[] {
  const requests: unknown[] = []
  for (const b of t.bands ?? [])
    requests.push({
      repeatCell: {
        range: { sheetId, startRowIndex: b.row, endRowIndex: b.row + 1 },
        cell: { userEnteredFormat: { backgroundColor: hex(b.color), textFormat: { bold: !!b.bold } } },
        fields: 'userEnteredFormat(backgroundColor,textFormat.bold)',
      },
    })
  for (const c of t.boldCols ?? [])
    requests.push({
      repeatCell: {
        range: { sheetId, startColumnIndex: c, endColumnIndex: c + 1 },
        cell: { userEnteredFormat: { textFormat: { bold: true } } },
        fields: 'userEnteredFormat.textFormat.bold',
      },
    })
  for (const f of t.fills ?? [])
    requests.push({
      repeatCell: {
        range: { sheetId, startRowIndex: f.row, endRowIndex: f.row + 1, startColumnIndex: f.col, endColumnIndex: f.col + (f.cols ?? 1) },
        cell: {
          userEnteredFormat: {
            ...(f.bg && { backgroundColor: hex(f.bg) }),
            textFormat: { bold: !!f.bold, ...(f.fg && { foregroundColor: hex(f.fg) }) },
          },
        },
        fields: `userEnteredFormat(${f.bg ? 'backgroundColor,' : ''}textFormat)`,
      },
    })
  const widths = t.colWidths ?? t.widths?.map((w) => Math.round(w * 8 + 16))
  widths?.forEach((px, c) =>
    requests.push({
      updateDimensionProperties: {
        range: { sheetId, dimension: 'COLUMNS', startIndex: c, endIndex: c + 1 },
        properties: { pixelSize: px },
        fields: 'pixelSize',
      },
    }),
  )
  return requests
}

const stamp = () => {
  const d = new Date()
  const z = (n: number) => String(n).padStart(2, '0')
  return `${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}`
}

const sheetsApi = (id: string, path = '') => `https://sheets.googleapis.com/v4/spreadsheets/${encodeURIComponent(id)}${path}`
const post = (url: string, body: unknown) =>
  api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

async function writeValues(spreadsheetId: string, sheets: { name: string; rows: Cell[][] }[]) {
  await post(sheetsApi(spreadsheetId, '/values:batchUpdate'), {
    valueInputOption: 'RAW',
    data: sheets.map((t) => ({ range: `'${t.name.replace(/'/g, "''")}'!A1`, values: t.rows.map((r) => r.map((c) => c ?? '')) })),
  })
}

const hex = (h: string) => {
  const n = parseInt(h.replace('#', ''), 16)
  return { red: ((n >> 16) & 255) / 255, green: ((n >> 8) & 255) / 255, blue: (n & 255) / 255 }
}

// 新しく作るスプレッドシートのロケール（小数点・日付・関数の区切りに効く）。
// スペイン語は中南米の語彙に合わせてメキシコ（es_419 はスプレッドシートのロケールに無い）
const SHEETS_LOCALE: Record<AppLanguage, string> = { ja: 'ja_JP', en: 'en_US', ko: 'ko_KR', es: 'es_MX', de: 'de_DE', 'pt-BR': 'pt_BR' }
export const spreadsheetLocale = (lang: string | undefined) => (isAppLanguage(lang) ? SHEETS_LOCALE[lang] : 'ja_JP')

/** 書式付きの新しいスプレッドシートを作成する（ロケールは setGoogleMessages で渡した表示言語） */
export async function createSpreadsheet(title: string, sheets: SheetSpec[]): Promise<GoogleFile> {
  const res = await post('https://sheets.googleapis.com/v4/spreadsheets', {
    properties: { title, locale: spreadsheetLocale(MSG.locale) },
    sheets: sheets.map((t, i) => ({
      properties: { sheetId: i, title: t.name, gridProperties: { frozenRowCount: t.frozenRows ?? 0, frozenColumnCount: t.frozenCols ?? 0 } },
    })),
  })
  const json = await res.json()
  const id: string = json.spreadsheetId
  await writeValues(id, sheets)

  const requests = sheets.flatMap((t, sheetId) => formatRequests(sheetId, t))
  if (requests.length) await post(sheetsApi(id, ':batchUpdate'), { requests })
  return { id, name: title, mimeType: SHEET_MIME, url: json.spreadsheetUrl }
}

/**
 * 結果を書き出す。元が Google スプレッドシートならタブを追加し、
 * そうでなければ（アップロードした Excel・Drive 上の .xlsx）新しいスプレッドシートを作成する。
 */
export async function writeResults(sheets: SheetSpec[], target: GoogleFile | null, title: string): Promise<string> {
  const suffix = stamp()
  const tabs = sheets.map((s) => ({ ...s, name: `${s.name}_${suffix}` }))
  if (target && target.mimeType === SHEET_MIME) {
    const res = await post(sheetsApi(target.id, ':batchUpdate'), { requests: tabs.map((t) => ({ addSheet: { properties: { title: t.name } } })) })
    const ids: number[] = (await res.json()).replies.map((r: any) => r.addSheet.properties.sheetId)
    await writeValues(target.id, tabs)
    const requests = tabs.flatMap((t, i) => formatRequests(ids[i], t))
    if (requests.length) await post(sheetsApi(target.id, ':batchUpdate'), { requests })
    return `https://docs.google.com/spreadsheets/d/${target.id}/edit`
  }
  return (await createSpreadsheet(title, tabs)).url
}
