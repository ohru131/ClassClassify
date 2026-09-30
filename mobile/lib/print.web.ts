// expo-print の Web 実装は渡した HTML ではなく「今の画面」を印刷してしまうので使わない。
// 新しいウィンドウに HTML を書いて、そのウィンドウで印刷する（PDF 保存はブラウザの印刷画面から）。
export async function printHtml(html: string): Promise<void> {
  const w = window.open('', '_blank')
  if (!w) throw new Error('ポップアップがブロックされました。ブラウザの設定で許可してください。')
  w.document.open()
  w.document.write(html)
  w.document.close()
  w.focus()
  setTimeout(() => w.print(), 300)
}

export async function sharePdf(html: string): Promise<void> {
  await printHtml(html)
}

export const canSharePdf = false
