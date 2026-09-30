// Web はダウンロードとブラウザの印刷画面なので、アプリが持つファイルは無い
export async function exportUri(fileName: string): Promise<string> {
  return fileName
}
export async function removeFile(_uri: string | null | undefined): Promise<void> {}
export async function clearAppFiles(): Promise<void> {}
