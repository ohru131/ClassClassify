// xlsx を読み込まない軽い部品（Web 版の最初の画面からも使う）

/** ファイル名に使えない文字・空白を _ にする（Web 版・スマホ版で共通） */
export const safeFileName = (s: string) => s.replace(/[\\/:*?"<>|\s]+/g, '_').replace(/^_+|_+$/g, '') || 'file'
