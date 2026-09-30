import { useMemo, useState } from 'react'
import { Modal, Pressable, ScrollView, Text, TextInput, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import { useI18n } from '@/lib/language-provider'
import { useLayout } from '@/lib/layout'
import { roster, type Problem } from '@/lib/solver'
import { StudentEditor } from './student-editor'
import { C } from './theme'
import { Btn, Card, styles } from './ui'

type Props = { problem: Problem; onChange: (f: (p: Problem) => Problem) => void }

const matches = (p: Problem, i: number, q: string) => {
  if (!q) return true
  const s = p.students[i]
  return String(s.no) === q || s.name.includes(q) || Object.values(s.values).some((v) => v === q)
}

/** 生徒一覧。狭い画面はカードの列＋編集はモーダル、広い画面は表（横スクロール）＋右に編集欄 */
export function StudentList({ problem, onChange }: Props) {
  const { isWide } = useLayout()
  const { t } = useI18n()
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const q = query.trim()
  const visible = useMemo(() => problem.students.map((_, i) => i).filter((i) => matches(problem, i, q)), [problem, q])
  const sel = selected !== null && selected < problem.students.length ? selected : null

  const add = () => {
    onChange((p) => roster.addStudent(p))
    setSelected(problem.students.length)
  }

  const header = (
    <View style={[styles.row, { flexWrap: 'wrap' }]}>
      <TextInput
        style={[styles.input, { flex: 1, minWidth: 180 }]}
        value={query}
        onChangeText={setQuery}
        placeholder={t('filterPlaceholder')}
        accessibilityLabel={t('filterPlaceholder')}
        returnKeyType="search"
      />
      <Btn variant="soft" icon="person-add-outline" label={t('addStudent')} onPress={add} />
    </View>
  )

  const editor =
    sel !== null ? <StudentEditor key={`${sel}-${problem.students[sel].no}`} problem={problem} index={sel} onChange={onChange} onClose={() => setSelected(null)} /> : null

  if (isWide) {
    return (
      <View style={{ flexDirection: 'row', gap: 14, alignItems: 'flex-start' }}>
        <Card style={{ flex: 3, gap: 12, minWidth: 0 }}>
          {header}
          <Text style={{ fontSize: 12, color: C.muted }}>
            {t('countWide', { shown: visible.length, total: problem.students.length })}
          </Text>
          <RosterTable problem={problem} rows={visible} selected={sel} onSelect={setSelected} />
        </Card>
        <Card style={{ flex: 2, minWidth: 300 }}>{editor ?? <Text style={{ color: C.muted }}>{t('selectToEdit')}</Text>}</Card>
      </View>
    )
  }

  return (
    <Card style={{ gap: 10 }}>
      {header}
      <Text style={{ fontSize: 12, color: C.muted }}>
        {t('countNarrow', { shown: visible.length, total: problem.students.length })}
      </Text>
      {visible.map((i) => (
        <StudentRow key={i} problem={problem} index={i} onPress={() => setSelected(i)} />
      ))}
      <Modal visible={sel !== null} animationType="slide" presentationStyle="pageSheet" onRequestClose={() => setSelected(null)}>
        <SafeAreaView style={{ flex: 1, backgroundColor: C.bg }}>
          <ScrollView contentContainerStyle={{ padding: 16 }} keyboardShouldPersistTaps="handled">
            {editor}
          </ScrollView>
        </SafeAreaView>
      </Modal>
    </Card>
  )
}

function StudentRow({ problem, index, onPress }: { problem: Problem; index: number; onPress: () => void }) {
  const { t } = useI18n()
  const s = problem.students[index]
  const tags = problem.columns.filter((c) => s.values[c.name]).map((c) => (c.kind === 'flag' ? c.name : `${c.name}:${s.values[c.name]}`))
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={t('editStudentA11y', { no: s.no, name: s.name })}
      onPress={onPress}
      style={(st: { pressed: boolean; hovered?: boolean }) => [
        { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 10, paddingHorizontal: 8, borderRadius: 12, borderBottomWidth: 1, borderColor: '#F1F5F9' },
        (st.pressed || st.hovered) && { backgroundColor: C.hover },
      ]}
    >
      <Text style={{ width: 34, fontSize: 13, color: C.muted, fontVariant: ['tabular-nums'] }}>{s.no}</Text>
      <View style={{ flex: 1 }}>
        <Text style={{ fontSize: 15, fontWeight: '700', color: s.name ? C.text : C.muted }}>{s.name || t('noName')}</Text>
        {tags.length ? (
          <Text style={{ fontSize: 12, color: C.sub, marginTop: 2 }} numberOfLines={1}>
            {tags.join(' · ')}
          </Text>
        ) : null}
      </View>
      <Text style={{ color: C.muted, fontSize: 18 }}>›</Text>
    </Pressable>
  )
}

/** 広い画面向けの表。列が多い名簿は横にスクロールする */
function RosterTable({ problem, rows, selected, onSelect }: { problem: Problem; rows: number[]; selected: number | null; onSelect: (i: number) => void }) {
  const { t } = useI18n()
  const cols = problem.columns
  const cell = (w: number) => ({ width: w, paddingHorizontal: 8, paddingVertical: 8 })
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator>
      <View>
        <View style={{ flexDirection: 'row', backgroundColor: '#F8FAFC', borderRadius: 8 }}>
          <Text style={[cell(56), th]}>{t('colNo')}</Text>
          <Text style={[cell(130), th]}>{t('colName')}</Text>
          {cols.map((c) => (
            <Text key={c.name} style={[cell(96), th]} numberOfLines={1}>
              {c.name}
            </Text>
          ))}
        </View>
        {rows.map((i) => {
          const s = problem.students[i]
          return (
            <Pressable
              key={i}
              accessibilityRole="button"
              accessibilityLabel={t('editStudentA11y', { no: s.no, name: s.name })}
              accessibilityState={{ selected: selected === i }}
              onPress={() => onSelect(i)}
              style={(st: { pressed: boolean; hovered?: boolean; focused?: boolean }) => [
                { flexDirection: 'row', borderBottomWidth: 1, borderColor: '#F1F5F9' },
                selected === i ? { backgroundColor: C.primarySoft } : (st.hovered || st.pressed || st.focused) && { backgroundColor: C.hover },
              ]}
            >
              <Text style={[cell(56), td, { color: C.muted }]}>{s.no}</Text>
              <Text style={[cell(130), td, { fontWeight: '700' }]} numberOfLines={1}>
                {s.name || t('noName')}
              </Text>
              {cols.map((c) => (
                <Text key={c.name} style={[cell(96), td]} numberOfLines={1}>
                  {s.values[c.name] ?? ''}
                </Text>
              ))}
            </Pressable>
          )
        })}
      </View>
    </ScrollView>
  )
}

const th = { fontSize: 12, fontWeight: '800' as const, color: C.sub }
const td = { fontSize: 14, color: C.text }
