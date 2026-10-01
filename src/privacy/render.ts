import { COPY } from '../../mobile/lib/copy'
import { PRIVACY, PRIVACY_SCOPE, PRIVACY_UPDATED, PRIVACY_UPDATED_LABEL } from '../../mobile/lib/copy/privacy'
import { APP_LANGUAGES, DEFAULT_LANGUAGE, LANGUAGE_META, type AppLanguage } from '../i18n/languages'

// 公開用プライバシーポリシー（https://ohru131.github.io/ClassClassify/privacy/）の本文を組み立てる純関数。
// **本文の情報源はスマホ版の mobile/lib/copy/privacy.ts だけ**で、ここに文章を書かない
// （アプリ内の画面とストアに登録する URL の中身が食い違うと、審査でもデータセーフティでも説明が付かない）。
// ビルド時に vite.config.ts のプラグインが privacy/index.html の <!--PRIVACY_CONTENT--> へ埋め込むので、
// JavaScript を読まないクローラーにも6言語ぶんの本文がそのまま見える。

const escapeHtml = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!)

/** 本文中の URL だけをリンクにする（それ以外はエスケープしたまま） */
export function linkify(text: string): string {
  return text
    .split(/(https?:\/\/[^\s)）]+)/)
    .map((part, i) => (i % 2 === 1 ? `<a href="${escapeHtml(part)}" rel="noopener">${escapeHtml(part)}</a>` : escapeHtml(part)))
    .join('')
}

export const privacyTitle = (lang: AppLanguage) => COPY[lang].privacyTitle

function renderSection(lang: AppLanguage): string {
  const sections = PRIVACY[lang]
    .map((s) => `<section><h2>${escapeHtml(s.title)}</h2>${s.body.map((b) => `<p>${linkify(b)}</p>`).join('')}</section>`)
    .join('')
  // 既定（英語）以外は hidden。JavaScript が動けば選んだ言語だけを出し、動かなければ英語が見える
  const hidden = lang === DEFAULT_LANGUAGE ? '' : ' hidden'
  return (
    `<article class="policy" lang="${LANGUAGE_META[lang].intl}" data-lang="${lang}"${hidden}>` +
    `<h1>${escapeHtml(privacyTitle(lang))}</h1>` +
    `<p class="meta">Mosaic · ${escapeHtml(PRIVACY_UPDATED_LABEL[lang])}: <time datetime="${PRIVACY_UPDATED}">${PRIVACY_UPDATED}</time></p>` +
    `<p class="scope">${escapeHtml(PRIVACY_SCOPE[lang])}</p>` +
    sections +
    `</article>`
  )
}

/** 言語の切り替え（?lang= のリンク。JavaScript が無くても動く）と6言語ぶんの本文 */
export function renderPrivacyContent(): string {
  const nav = APP_LANGUAGES.map((l) => `<a href="?lang=${l}" data-lang-link="${l}" hreflang="${LANGUAGE_META[l].intl}">${escapeHtml(LANGUAGE_META[l].endonym)}</a>`).join('')
  return `<nav class="langs" aria-label="Language">${nav}</nav>` + APP_LANGUAGES.map(renderSection).join('')
}
