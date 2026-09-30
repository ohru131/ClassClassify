/**
 * Google スプレッドシート連携（Google Identity Services + Google Picker + Drive/Sheets API）。
 * スコープは drive.file のみ: ユーザーが Picker で選んだファイルと、このアプリが作成したファイルにだけアクセスできる。
 * 通信はブラウザ ⇔ Google の間だけで行い、第三者のサーバーは経由しない。
 */

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined
const API_KEY = import.meta.env.VITE_GOOGLE_API_KEY as string | undefined
/** Google Cloud プロジェクト番号。Picker で選んだファイルへのアクセス権付与に必要 */
const APP_ID = import.meta.env.VITE_GOOGLE_APP_ID as string | undefined

const SCOPE = 'https://www.googleapis.com/auth/drive.file'
const SHEET_MIME = 'application/vnd.google-apps.spreadsheet'
const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

export const googleEnabled = !!(CLIENT_ID && API_KEY && APP_ID)

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
        reject(new Error(`${src} を読み込めませんでした`))
      }
      document.head.appendChild(s)
    })
    loaded.set(src, p)
  }
  return p
}

let token: { value: string; expires: number } | null = null

async function getToken(): Promise<string> {
  if (token && token.expires > Date.now() + 60_000) return token.value
  await loadScript('https://accounts.google.com/gsi/client')
  return new Promise((resolve, reject) => {
    const client = window.google.accounts.oauth2.initTokenClient({
      client_id: CLIENT_ID,
      scope: SCOPE,
      callback: (res: any) => {
        if (res.error) return reject(new Error(`Google 認証に失敗しました: ${res.error_description ?? res.error}`))
        token = { value: res.access_token, expires: Date.now() + Number(res.expires_in) * 1000 }
        resolve(res.access_token)
      },
      error_callback: (err: any) =>
        reject(new Error(err?.type === 'popup_closed' ? 'cancelled' : `Google 認証に失敗しました: ${err?.message ?? err?.type}`)),
    })
    client.requestAccessToken({ prompt: token ? '' : undefined })
  })
}

async function api(url: string, init: RequestInit = {}): Promise<Response> {
  const t = await getToken()
  const res = await fetch(url, { ...init, headers: { ...init.headers, Authorization: `Bearer ${t}` } })
  if (!res.ok) {
    let msg = `${res.status}`
    try {
      msg = (await res.json()).error?.message ?? msg
    } catch {
      /* ignore */
    }
    throw new Error(`Google API エラー: ${msg}`)
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
      .setLocale('ja')
      .setTitle('名簿のスプレッドシートを選択')
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
}

const stamp = () => {
  const d = new Date()
  const z = (n: number) => String(n).padStart(2, '0')
  return `${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}`
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

/** 書式付きの新しいスプレッドシートを作成する */
export async function createSpreadsheet(title: string, sheets: SheetSpec[]): Promise<GoogleFile> {
  const res = await post('https://sheets.googleapis.com/v4/spreadsheets', {
    properties: { title, locale: 'ja_JP' },
    sheets: sheets.map((t, i) => ({
      properties: { sheetId: i, title: t.name, gridProperties: { frozenRowCount: t.frozenRows ?? 0, frozenColumnCount: t.frozenCols ?? 0 } },
    })),
  })
  const json = await res.json()
  const id: string = json.spreadsheetId
  await writeValues(id, sheets)

  const requests: unknown[] = []
  sheets.forEach((t, sheetId) => {
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
    t.colWidths?.forEach((px, c) =>
      requests.push({
        updateDimensionProperties: {
          range: { sheetId, dimension: 'COLUMNS', startIndex: c, endIndex: c + 1 },
          properties: { pixelSize: px },
          fields: 'pixelSize',
        },
      }),
    )
  })
  if (requests.length) await post(sheetsApi(id, ':batchUpdate'), { requests })
  return { id, name: title, mimeType: SHEET_MIME, url: json.spreadsheetUrl }
}

/**
 * 結果を書き出す。元が Google スプレッドシートならタブを追加し、
 * そうでなければ（アップロードした Excel・Drive 上の .xlsx）新しいスプレッドシートを作成する。
 */
export async function writeResults(sheets: { name: string; rows: Cell[][] }[], target: GoogleFile | null, title: string): Promise<string> {
  const suffix = stamp()
  const tabs = sheets.map((s) => ({ ...s, name: `${s.name}_${suffix}` }))
  if (target && target.mimeType === SHEET_MIME) {
    await post(sheetsApi(target.id, ':batchUpdate'), { requests: tabs.map((t) => ({ addSheet: { properties: { title: t.name } } })) })
    await writeValues(target.id, tabs)
    return `https://docs.google.com/spreadsheets/d/${target.id}/edit`
  }
  return (await createSpreadsheet(title, tabs)).url
}
