import { useState } from 'react'
import { Text, TextInput, View } from 'react-native'

import { confirmAction } from '@/lib/confirm'
import { useI18n } from '@/lib/language-provider'
import { roster, type ColumnKind, type ColumnSpec, type Problem } from '@/lib/solver'
import { C } from './theme'
import { Btn, Card, Segmented, Stepper, styles } from './ui'

const KIND_KEY = { flag: 'kindFlag', category: 'kindCategory', numeric: 'kindNumeric' } as const

const levelsText = (c: ColumnSpec, blankOnly: string) =>
  c.levels.length === 0 ? blankOnly : c.kind === 'numeric' ? `${c.levels[0]}–${c.levels[c.levels.length - 1]}` : c.levels.join(' / ')

/** 項目（特性）の一覧・重み・追加・削除 */
export function ColumnEditor({ problem, onChange }: { problem: Problem; onChange: (f: (p: Problem) => Problem) => void }) {
  const { t, file, num } = useI18n()
  const i18nNum = (v: number) => num(v, 1)
  const [name, setName] = useState('')
  const [kind, setKind] = useState<ColumnKind>('flag')
  const trimmed = name.trim()
  const duplicate = problem.columns.some((c) => c.name === trimmed) || ['NO', '名前', file.no, file.name].includes(trimmed)
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
          {t('columnsHelp')}
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
                  {c.kind === 'category' ? t(KIND_KEY[c.kind]) : `${t(KIND_KEY[c.kind])} · ${levelsText(c, t('blankOnly'))}`}
                </Text>
              </View>
              <Btn
                small
                variant="danger"
                icon="trash-outline"
                accessibilityLabel={t('deleteColumnA11y', { name: c.name })}
                onPress={async () => {
                  if (await confirmAction(t('deleteColumnTitle'), t('deleteColumnBody', { name: c.name }), t('delete'), t('cancel'))) onChange((p) => roster.removeColumn(p, c.name))
                }}
              />
            </View>
            <View style={[styles.row, { justifyContent: 'space-between' }]}>
              <Text style={styles.label}>{t('weight')}</Text>
              <Stepper label={t('weightA11y', { name: c.name })} value={w} min={0} max={5} step={0.5} format={(v) => i18nNum(v)} onChange={(v) => setWeight(i, v)} />
            </View>
            {c.kind === 'category' ? <OptionsEditor problem={problem} column={c} onChange={onChange} /> : null}
            {c.levels.length === 0 && c.kind !== 'category' ? <Text style={{ fontSize: 12, color: C.warn }}>{t('columnEmpty')}</Text> : null}
          </Card>
        )
      })}
      <Card style={{ gap: 10 }}>
        <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>{t('addColumnTitle')}</Text>
        <TextInput
          style={styles.input}
          value={name}
          onChangeText={setName}
          onSubmitEditing={add}
          returnKeyType="done"
          placeholder={t('addColumnPlaceholder')}
          accessibilityLabel={t('addColumnTitle')}
        />
        <Segmented value={kind} onChange={setKind} options={(['flag', 'category', 'numeric'] as const).map((k) => ({ value: k, label: t(KIND_KEY[k]) }))} />
        {duplicate && trimmed ? <Text style={{ color: C.danger, fontSize: 12 }}>{t('duplicateColumn')}</Text> : null}
        <Btn variant="soft" icon="add" label={t('add')} disabled={!trimmed || duplicate} onPress={add} />
      </Card>
    </View>
  )
}

/** リストの選択肢の一覧・追加・名前の変更・削除 */
function OptionsEditor({ problem, column, onChange }: { problem: Problem; column: ColumnSpec; onChange: (f: (p: Problem) => Problem) => void }) {
  const { t } = useI18n()
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [added, setAdded] = useState('')
  const name = column.name
  const count = (l: string) => problem.students.filter((s) => s.values[name] === l).length
  const v = added.trim()
  const addDup = column.levels.includes(v)
  const add = () => {
    if (!v || addDup) return
    onChange((p) => roster.addLevel(p, name, v))
    setAdded('')
  }
  const d = draft.trim()
  const renameDup = editing !== null && d !== editing && column.levels.includes(d)
  const rename = () => {
    if (editing !== null && d && !renameDup) onChange((p) => roster.renameLevel(p, name, editing, d))
    setEditing(null)
  }
  return (
    <View style={{ gap: 8 }}>
      <Text style={styles.label}>{t('listOptions')}</Text>
      {column.levels.map((l) =>
        editing === l ? (
          <View key={l} style={styles.row}>
            <TextInput style={[styles.input, { flex: 1 }]} value={draft} onChangeText={setDraft} onSubmitEditing={rename} autoFocus returnKeyType="done" accessibilityLabel={t('editOptionA11y', { v: l })} />
            <Btn small variant="soft" icon="checkmark" accessibilityLabel={t('save')} disabled={!d || renameDup} onPress={rename} />
            <Btn small icon="close" accessibilityLabel={t('cancel')} onPress={() => setEditing(null)} />
          </View>
        ) : (
          <View key={l} style={[styles.row, { paddingLeft: 12, borderRadius: 10, backgroundColor: '#F8FAFC' }]}>
            <Text style={{ flex: 1, fontSize: 15, fontWeight: '700', color: C.text }}>{l}</Text>
            <Text style={{ fontSize: 12, color: C.muted, fontVariant: ['tabular-nums'] }}>{count(l)}</Text>
            <Btn
              small
              icon="create-outline"
              accessibilityLabel={t('editOptionA11y', { v: l })}
              onPress={() => {
                setEditing(l)
                setDraft(l)
              }}
              style={{ backgroundColor: 'transparent', borderWidth: 0 }}
            />
            <Btn
              small
              icon="trash-outline"
              variant="danger"
              accessibilityLabel={t('deleteOptionA11y', { v: l })}
              onPress={async () => {
                const n = count(l)
                if (n === 0 || (await confirmAction(t('deleteOptionTitle'), t('deleteOptionBody', { v: l, n }), t('delete'), t('cancel')))) onChange((p) => roster.removeLevel(p, name, l))
              }}
              style={{ backgroundColor: 'transparent' }}
            />
          </View>
        ),
      )}
      {renameDup ? <Text style={{ color: C.danger, fontSize: 12 }}>{t('duplicateOption')}</Text> : null}
      <View style={styles.row}>
        <TextInput
          style={[styles.input, { flex: 1 }]}
          value={added}
          onChangeText={setAdded}
          onSubmitEditing={add}
          returnKeyType="done"
          placeholder={t('newValuePlaceholder')}
          placeholderTextColor={C.muted}
          accessibilityLabel={t('addOptionChip')}
        />
        <Btn small variant="soft" icon="add" accessibilityLabel={t('addOptionChip')} disabled={!v || addDup} onPress={add} />
      </View>
      {addDup && v ? <Text style={{ color: C.danger, fontSize: 12 }}>{t('duplicateOption')}</Text> : null}
      <Text style={{ fontSize: 12, color: C.muted }}>{t('listOptionsHint')}</Text>
    </View>
  )
}
