package expo.modules.saffiles

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.util.Base64
import expo.modules.kotlin.Promise
import expo.modules.kotlin.exception.CodedException
import expo.modules.kotlin.exception.Exceptions
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.io.IOException

private const val OPEN_CODE = 48710
private const val CREATE_CODE = 48711
private const val XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

// OS の「開く」「保存」ダイアログ（Google ドライブも並ぶ）を直接使う。
// expo-document-picker は Google スプレッドシート（仮想ファイル）を開けず、保存ダイアログも無いため。
class SafFilesModule : Module() {
  private val context: Context
    get() = appContext.reactContext ?: throw Exceptions.ReactContextLost()

  private var pendingPromise: Promise? = null
  private var pendingBytes: ByteArray? = null

  override fun definition() = ModuleDefinition {
    Name("SafFiles")

    // 戻り値は { name, base64 }。キャンセルなら null
    AsyncFunction("openDocumentAsync") { mimeTypes: List<String>, promise: Promise ->
      if (!begin(promise)) return@AsyncFunction
      val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
        addCategory(Intent.CATEGORY_OPENABLE)
        type = "*/*"
        putExtra(Intent.EXTRA_MIME_TYPES, mimeTypes.toTypedArray())
      }
      appContext.throwingActivity.startActivityForResult(intent, OPEN_CODE)
    }

    // 戻り値は保存したファイル名。キャンセルなら null
    AsyncFunction("createDocumentAsync") { fileName: String, mimeType: String, base64: String, promise: Promise ->
      if (!begin(promise)) return@AsyncFunction
      pendingBytes = Base64.decode(base64, Base64.DEFAULT)
      val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
        addCategory(Intent.CATEGORY_OPENABLE)
        type = mimeType
        putExtra(Intent.EXTRA_TITLE, fileName)
      }
      appContext.throwingActivity.startActivityForResult(intent, CREATE_CODE)
    }

    OnActivityResult { _, (requestCode, resultCode, intent) ->
      if (requestCode != OPEN_CODE && requestCode != CREATE_CODE) return@OnActivityResult
      val promise = pendingPromise ?: return@OnActivityResult
      val bytes = pendingBytes
      pendingPromise = null
      pendingBytes = null

      val uri = intent?.data
      if (resultCode != Activity.RESULT_OK || uri == null) {
        promise.resolve(null)
        return@OnActivityResult
      }
      // ドライブへの入出力はネットワークを挟むので、メインスレッドから外す
      Thread {
        try {
          if (requestCode == OPEN_CODE) promise.resolve(read(uri)) else promise.resolve(write(uri, bytes ?: ByteArray(0)))
        } catch (e: Exception) {
          promise.reject(CodedException("ERR_SAF_FILES", e.message ?: "file access failed", e))
        }
      }.start()
    }
  }

  private fun begin(promise: Promise): Boolean {
    if (pendingPromise != null) {
      promise.reject(CodedException("ERR_SAF_BUSY", "A file dialog is already open", null))
      return false
    }
    pendingPromise = promise
    return true
  }

  private fun displayName(uri: Uri): String? =
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
      if (it.moveToFirst()) it.getString(0) else null
    }

  private fun isVirtual(uri: Uri): Boolean {
    val flags = context.contentResolver.query(uri, arrayOf(DocumentsContract.Document.COLUMN_FLAGS), null, null, null)?.use {
      if (it.moveToFirst()) it.getInt(0) else 0
    } ?: 0
    return flags and DocumentsContract.Document.FLAG_VIRTUAL_DOCUMENT != 0
  }

  private fun read(uri: Uri): Map<String, String> {
    val resolver = context.contentResolver
    // Google スプレッドシートなどの仮想ファイルは openInputStream では開けない。xlsx に変換して受け取る
    val stream = if (isVirtual(uri)) {
      resolver.openTypedAssetFileDescriptor(uri, XLSX_MIME, null)?.createInputStream()
    } else {
      resolver.openInputStream(uri)
    } ?: throw IOException("Could not open $uri")
    val bytes = stream.use { it.readBytes() }
    return mapOf("name" to (displayName(uri) ?: ""), "base64" to Base64.encodeToString(bytes, Base64.NO_WRAP))
  }

  private fun write(uri: Uri, bytes: ByteArray): String {
    val out = context.contentResolver.openOutputStream(uri, "wt") ?: throw IOException("Could not open $uri")
    out.use { it.write(bytes) }
    return displayName(uri) ?: ""
  }
}
