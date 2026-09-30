import { useState } from 'react'
import { Text, TextInput, View } from 'react-native'

import { confirmAction } from '@/lib/confirm'
import { roster, type ColumnKind, type ColumnSpec, type Problem } from '@/lib/solver'
import { C } from './theme'
import { Btn, Card, Segmented, Stepper, styles } from './ui'

const KIND_LABEL: Record<ColumnKind, string> = { flag: '該当', category: 'カテゴリ', numeric: '数値' }

const levelsText = (c: ColumnSpec) =>
  c.levels.length === 0 ? '（空欄のみ）' : c.kind === 'numeric' ? `${c.levels[0]}〜${c.levels[c.levels.length - 1]}` : c.levels.join(' / ')

/** 項目（特性）の一覧・重み・追加・削除 */
export function ColumnEditor({ problem, onChange }: { problem: Problem; onChange: (f: (p: Problem) => Problem) => void }) {
  const [name, setName] = useState('')
  const [kind, setKind] = useState<ColumnKind>('flag')
  const trimmed = name.trim()
  const duplicate = problem.columns.some((c) => c.name === trimmed) || trimmed === 'NO' || trimmed === '名前'
  const add = () => {
    if (!trimmed || duplicate) return
    onChange((p) => roster.addColumn(p, trimmed, kind))
    setName('')
  }
  const setWeight = (i: number, w: number) =>
    onChange((p) => ({ ...p, columns: p.columns.map((c, j) => (j === i ? { ...c, weight: w, enabled: w > 0 && c.levels.length > 0 } : c)) }))

  return (
    <View style={{ gap: 14 }}>
      <Card style={{ gap: 4 }}>
        <Text style={{ fontSize: 13, color: C.sub, lineHeight: 19 }}>
          重みが大きい項目ほど優先して各クラスへ均等に散らします。0 にするとその項目は無視します。値が1種類（○ と空欄）なら「該当」、数種類なら「カテゴリ」、7種類以上の数値は「数値」（平均を揃える）として扱います。
        </Text>
      </Card>
      {problem.columns.map((c, i) => {
        const w = c.enabled ? c.weight : 0
        return (
          <Card key={c.name} style={{ gap: 8, opacity: w > 0 ? 1 : 0.7 }}>
            <View style={[styles.row, { justifyContent: 'space-between' }]}>
              <View style={{ flex: 1 }}>
                <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>{c.name}</Text>
                <Text style={{ fontSize: 12, color: C.sub, marginTop: 2 }} numberOfLines={2}>
                  {KIND_LABEL[c.kind]} · {levelsText(c)}
                </Text>
              </View>
              <Btn
                small
                variant="danger"
                icon="trash-outline"
                accessibilityLabel={`項目「${c.name}」を削除`}
                onPress={async () => {
                  if (await confirmAction('項目を削除', `「${c.name}」を全員の名簿から削除します。`, '削除')) onChange((p) => roster.removeColumn(p, c.name))
                }}
              />
            </View>
            <View style={[styles.row, { justifyContent: 'space-between' }]}>
              <Text style={styles.label}>重み</Text>
              <Stepper label={`${c.name}の重み`} value={w} min={0} max={5} step={0.5} format={(v) => v.toFixed(1)} onChange={(v) => setWeight(i, v)} />
            </View>
            {c.levels.length === 0 ? <Text style={{ fontSize: 12, color: C.warn }}>全員が空欄のため、この項目は使われません。生徒の編集で値を入れてください。</Text> : null}
          </Card>
        )
      })}
      <Card style={{ gap: 10 }}>
        <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>項目を追加</Text>
        <TextInput
          style={styles.input}
          value={name}
          onChangeText={setName}
          onSubmitEditing={add}
          returnKeyType="done"
          placeholder="例: リーダー、ピアノ、通級"
          accessibilityLabel="追加する項目名"
        />
        <Segmented value={kind} onChange={setKind} options={(['flag', 'category', 'numeric'] as const).map((k) => ({ value: k, label: KIND_LABEL[k] }))} />
        {duplicate && trimmed ? <Text style={{ color: C.danger, fontSize: 12 }}>同じ名前の項目があります</Text> : null}
        <Btn variant="soft" icon="add" label="追加" disabled={!trimmed || duplicate} onPress={add} />
      </Card>
    </View>
  )
}
