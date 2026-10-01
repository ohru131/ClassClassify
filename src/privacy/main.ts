import { isAppLanguage, LANGUAGE_META, matchLanguageTag, resolveLanguageTags, type AppLanguage } from '../i18n/languages'
// src/copy/core.ts は読まない（Web 版の文言一式までこのページに入るため）
import { LANGUAGE_STORAGE_KEY } from '../i18n/storage-key'

// 公開用プライバシーポリシーの言語切り替え。本文はビルド時に HTML へ埋め込み済みなので、
// ここは「どの言語を見せるか」だけを決める（#ja などのフラグメント → ?lang= → Web 版で選んだ言語 → ブラウザの言語 → 英語）。
// JavaScript が動かないときは全言語が並んだまま見える（render.ts）。
const readStored = () => {
  try {
    return localStorage.getItem(LANGUAGE_STORAGE_KEY)
  } catch {
    return null
  }
}

function show(lang: AppLanguage) {
  document.querySelectorAll<HTMLElement>('[data-lang]').forEach((el) => {
    el.hidden = el.dataset.lang !== lang
  })
  document.querySelectorAll<HTMLAnchorElement>('[data-lang-link]').forEach((a) => {
    if (a.dataset.langLink === lang) a.setAttribute('aria-current', 'true')
    else a.removeAttribute('aria-current')
  })
  document.documentElement.lang = LANGUAGE_META[lang].intl
  const h1 = document.querySelector<HTMLElement>(`[data-lang="${lang}"] h1`)
  if (h1?.textContent) document.title = `${h1.textContent} — FairClass`
}

const stored = readStored()
const fromHash = window.location.hash.slice(1)
const initial =
  (isAppLanguage(fromHash) ? fromHash : null) ??
  matchLanguageTag(new URLSearchParams(window.location.search).get('lang')) ??
  (isAppLanguage(stored) ? stored : resolveLanguageTags(navigator.languages?.length ? navigator.languages : [navigator.language]))
show(initial)

document.querySelectorAll<HTMLAnchorElement>('[data-lang-link]').forEach((a) => {
  a.addEventListener('click', (e) => {
    const l = a.dataset.langLink
    if (!isAppLanguage(l)) return
    e.preventDefault()
    show(l)
    const url = new URL(window.location.href)
    url.searchParams.set('lang', l)
    url.hash = ''
    window.history.replaceState(null, '', url)
    window.scrollTo(0, 0)
  })
})
