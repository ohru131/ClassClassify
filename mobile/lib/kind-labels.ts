import type { ColumnKind } from './solver'

/** 項目の種類 → 画面の文言のキー（チェック・リスト・程度・数値） */
export const KIND_KEY = { flag: 'kindFlag', category: 'kindCategory', degree: 'kindDegree', numeric: 'kindNumeric' } as const satisfies Record<ColumnKind, string>
