import { useMemo, useState } from 'react'
import { Pressable, Text, TextInput, View } from 'react-native'

import { roster, UNWANTED_COLOR, wantedColor, type Problem } from '@/lib/solver'
import { useI18n } from '@/lib/language-provider'
import { useLayout } from '@/lib/layout'
import { C } from './theme'
import { Btn, Card, Notice, styles } from './ui'

type Kind = 'wanted' | 'unwanted'
const TITLE_KEY = { wanted: 'wantedTitle', unwanted: 'unwantedTitle' } as const
const nameOf = (p: Problem, i: number, no: string) => p.students[i].name || `${no} ${p.students[i].no}`

/** 「同じ組」「別の組」の指定。作成・編集はチェックリストで生徒を選ぶ（長押し・ドラッグ不要） */
export function GroupEditor({ problem, kind, onChange }: { problem: Problem; kind: Kind; onChange: (f: (p: Problem) => Problem) => void }) {
  const { t, file } = useI18n()
  const who = (i: number) => nameOf(problem, i, file.no)
  const prefix = file.tagPrefix[kind]
  const groups = kind === 'wanted' ? problem.wantedGroups : problem.unwantedGroups
  // editing: null=一覧、-1=新規、0以上=そのグループを編集
  const [editing, setEditing] = useState<number | null>(null)
  const conflicts = useMemo(() => roster.findConflicts(problem), [problem])

  if (editing !== null)
    return (
      <MemberPicker
        key={`${kind}${editing}`}
        problem={problem}
        title={editing < 0 ? t('addConditionTitle', { kind: t(TITLE_KEY[kind]) }) : t('editConditionTitle', { label: `${prefix}${editing + 1}` })}
        initial={editing >= 0 ? (groups[editing] ?? []) : []}
        onCancel={() => setEditing(null)}
        onSave={(members) => {
          onChange((p) => (editing < 0 ? roster.addGroup(p, kind, members) : roster.setGroup(p, kind, editing, members)))
          setEditing(null)
        }}
      />
    )

  return (
    <View style={{ gap: 12 }}>
      {conflicts.length ? (
        <Notice tone="error">
          <Text style={{ color: C.danger, fontSize: 13 }}>
            {t('conflict', { pairs: conflicts.map(([a, b]) => t('conflictPair', { a: who(a), b: who(b) })).join(file.listSep) })}
          </Text>
        </Notice>
      ) : null}
      <Card style={{ gap: 10 }}>
        <View style={[styles.row, { justifyContent: 'space-between', flexWrap: 'wrap' }]}>
          <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>
            {t(TITLE_KEY[kind])} <Text style={{ fontSize: 13, color: C.muted, fontWeight: '600' }}>{t('conditionsCount', { n: groups.length })}</Text>
          </Text>
          <Btn variant="soft" icon="add" label={t('addCondition')} onPress={() => setEditing(-1)} />
        </View>
        <Text style={{ fontSize: 12, color: C.sub }}>
          {t(kind === 'wanted' ? 'wantedHelp' : 'unwantedHelp')}
        </Text>
        {groups.length === 0 ? <Text style={{ color: C.muted }}>{t('noConditions')}</Text> : null}
        {groups.map((g, gi) => {
          const color = kind === 'wanted' ? wantedColor(gi) : UNWANTED_COLOR
          return (
            <View key={gi} style={[styles.row, { backgroundColor: color.bg, borderRadius: 12, padding: 10 }]}>
              <Text style={{ fontWeight: '800', color: color.fg, width: 40 }}>
                {prefix}
                {gi + 1}
              </Text>
              <Text style={{ flex: 1, color: C.text, fontSize: 14 }}>{g.map(who).join(file.joinSep)}</Text>
              <Btn small icon="create-outline" accessibilityLabel={t('edit')} onPress={() => setEditing(gi)} />
              <Btn small variant="danger" icon="trash-outline" accessibilityLabel={t('delete')} onPress={() => onChange((p) => roster.removeGroup(p, kind, gi))} />
            </View>
          )
        })}
      </Card>
    </View>
  )
}

function MemberPicker({ problem, title, initial, onSave, onCancel }: { problem: Problem; title: string; initial: number[]; onSave: (m: number[]) => void; onCancel: () => void }) {
  const [picked, setPicked] = useState<number[]>(initial)
  const [query, setQuery] = useState('')
  const { isWide } = useLayout()
  const { t, file } = useI18n()
  const q = query.trim()
  const rows = problem.students.map((_, i) => i).filter((i) => !q || problem.students[i].name.includes(q) || String(problem.students[i].no) === q)
  const toggle = (i: number) => setPicked((p) => (p.includes(i) ? p.filter((x) => x !== i) : [...p, i]))
  return (
    <Card style={{ gap: 10 }}>
      <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>{title}</Text>
      <Text style={{ fontSize: 13, color: C.sub }}>
        {t('selectedNames', { names: picked.length ? picked.map((i) => nameOf(problem, i, file.no)).join(file.joinSep) : t('selectedNone') })}
      </Text>
      <View style={[styles.row, { flexWrap: 'wrap' }]}>
        <Btn variant="primary" icon="checkmark" label={t('save')} disabled={picked.length < 2 && initial.length === 0} onPress={() => onSave(picked)} />
        <Btn label={t('cancel')} onPress={onCancel} />
        {initial.length > 0 && picked.length < 2 ? <Text style={{ fontSize: 12, color: C.warn }}>{t('willDelete')}</Text> : null}
      </View>
      <TextInput style={styles.input} value={query} onChangeText={setQuery} placeholder={t('filterNames')} accessibilityLabel={t('filterNames')} />
      <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 6 }}>
        {rows.map((i) => {
          const on = picked.includes(i)
          const s = problem.students[i]
          return (
            <Pressable
              key={i}
              accessibilityRole="checkbox"
              accessibilityState={{ checked: on }}
              accessibilityLabel={`${s.no} ${s.name}`}
              onPress={() => toggle(i)}
              style={(st: { pressed: boolean; hovered?: boolean }) => [
                {
                  width: isWide ? '32%' : '48%',
                  flexDirection: 'row',
                  alignItems: 'center',
                  gap: 8,
                  minHeight: 44,
                  paddingHorizontal: 10,
                  borderRadius: 10,
                  borderWidth: 1,
                  borderColor: on ? C.primary : C.border,
                  backgroundColor: on ? C.primarySoft : st.hovered || st.pressed ? C.hover : '#fff',
                },
              ]}
            >
              <Text style={{ fontSize: 16, color: on ? C.primary : C.muted }}>{on ? '☑' : '☐'}</Text>
              <Text style={{ fontSize: 12, color: C.muted, width: 26 }}>{s.no}</Text>
              <Text style={{ flex: 1, fontSize: 14, color: C.text }} numberOfLines={1}>
                {s.name || t('noName')}
              </Text>
            </Pressable>
          )
        })}
      </View>
    </Card>
  )
}
