// expo-print の Web 実装は渡した HTML ではなく「今の画面」を印刷してしまうので使わない。
// 新しいウィンドウに HTML を書いて、そのウィンドウで印刷する（PDF 保存はブラウザの印刷画面から）。
export async function printHtml(html: string, messages?: { popupBlocked: string }): Promise<void> {
  const w = window.open('', '_blank')
  if (!w) throw new Error(messages?.popupBlocked ?? 'Pop-up blocked')
  w.document.open()
  w.document.write(html)
  w.document.close()
  w.focus()
  setTimeout(() => w.print(), 300)
}

export async function sharePdf(html: string, _fileName?: string, messages?: { popupBlocked: string }): Promise<void> {
  await printHtml(html, messages)
}

/** Web（動作確認用）は保存ダイアログを持たない。PDF はブラウザの印刷画面から保存する */
export const savePdf = async (_html: string, _fileName: string): Promise<boolean> => false

export const canSharePdf = false
