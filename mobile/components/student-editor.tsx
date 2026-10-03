import Ionicons from '@expo/vector-icons/Ionicons'
import { useRef, useState } from 'react'
import { Pressable, Text, TextInput, View } from 'react-native'

import { confirmAction } from '@/lib/confirm'
import { useI18n } from '@/lib/language-provider'
import { DEGREE_LEVELS, roster, type ColumnSpec, type Problem } from '@/lib/solver'
import { C } from './theme'
import { Btn, Chip, styles } from './ui'

const KIND_KEY = { flag: 'kindFlag', category: 'kindCategory', degree: 'kindDegree', numeric: 'kindNumeric' } as const

/**
 * 生徒1人の編集。テキストは入力中は手元に持ち、確定（Enter・フォーカスが外れたとき）で名簿へ反映する
 * （1文字ごとに項目の種類を判定し直すと、入力の途中で欄の種類が変わってしまうため）。
 * 呼び出し側で key に生徒の index を渡し、別の生徒に切り替えたら作り直すこと。
 */
export function StudentEditor({ problem, index, onChange, onClose }: { problem: Problem; index: number; onChange: (f: (p: Problem) => Problem) => void; onClose?: () => void }) {
  const { t, file, lang } = useI18n()
  const s = problem.students[index]
  const [no, setNo] = useState(String(s.no))
  const [name, setName] = useState(s.name)
  const [noError, setNoError] = useState<string | null>(null)
  const nameRef = useRef<TextInput>(null)
  const groups = roster.groupsOf(problem)

  const commitNo = () => {
    const v = Number(no)
    if (!Number.isInteger(v) || v <= 0) return setNoError(t('noInvalid'))
    if (roster.isNoTaken(problem, v, index)) return setNoError(t('noTaken', { no: v }))
    setNoError(null)
    if (v !== s.no) onChange((p) => roster.updateStudent(p, index, { no: v }))
  }
  const commitName = () => {
    if (name !== s.name) onChange((p) => roster.updateStudent(p, index, { name }))
  }
  const setValue = (column: string, value: string) => onChange((p) => roster.updateStudent(p, index, { values: { [column]: value } }))

  return (
    <View style={{ gap: 14 }}>
      <View style={[styles.row, { justifyContent: 'space-between' }]}>
        <Text style={{ fontSize: 18, fontWeight: '800', color: C.text }}>{t('editStudent')}</Text>
        {onClose ? <Btn small icon="close" accessibilityLabel={t('close')} onPress={onClose} /> : null}
      </View>
      <View style={styles.row}>
        <View style={{ width: 96 }}>
          <Text style={styles.label}>{t('colNo')}</Text>
          <TextInput
            style={[styles.input, noError ? { borderColor: C.danger } : null]}
            value={no}
            onChangeText={setNo}
            onBlur={commitNo}
            onSubmitEditing={() => {
              commitNo()
              nameRef.current?.focus()
            }}
            keyboardType="number-pad"
            returnKeyType="next"
            accessibilityLabel={t('colNo')}
          />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.label}>{t('colName')}</Text>
          <TextInput
            ref={nameRef}
            style={styles.input}
            value={name}
            onChangeText={setName}
            onBlur={commitName}
            onSubmitEditing={commitName}
            returnKeyType="done"
            placeholder={t('namePlaceholder')}
            accessibilityLabel={t('colName')}
          />
        </View>
      </View>
      {noError ? <Text style={{ color: C.danger, fontSize: 12 }}>{noError}</Text> : null}

      {problem.columns.map((c) => (
        <ValueField key={c.name} column={c} value={s.values[c.name] ?? ''} onCommit={(v) => setValue(c.name, v)} onAddOption={(v) => onChange((p) => roster.addLevel(p, c.name, v))} flagMark={lang === 'ja' ? '○' : '✓'} />
      ))}
      {problem.columns.length === 0 ? <Text style={{ color: C.muted }}>{t('noColumns')}</Text> : null}

      {groups.wanted[index].length + groups.unwanted[index].length > 0 ? (
        <View>
          <Text style={styles.label}>{t('pairingsLabel')}</Text>
          <View style={styles.wrap}>
            {groups.wanted[index].map((g) => (
              <Chip key={`w${g}`} label={`${file.tagPrefix.wanted}${g + 1}: ${problem.wantedGroups[g].map((i) => problem.students[i].name || problem.students[i].no).join(file.joinSep)}`} bg="#E0F2FE" fg="#0369A1" />
            ))}
            {groups.unwanted[index].map((g) => (
              <Chip key={`u${g}`} label={`${file.tagPrefix.unwanted}${g + 1}: ${problem.unwantedGroups[g].map((i) => problem.students[i].name || problem.students[i].no).join(file.joinSep)}`} bg="#FFE4E6" fg="#BE123C" />
            ))}
          </View>
        </View>
      ) : null}

      <Btn
        variant="danger"
        icon="trash-outline"
        label={t('deleteStudent')}
        onPress={async () => {
          if (await confirmAction(t('deleteStudentTitle'), t('deleteStudentBody', { who: `${s.no} ${s.name || t('noName')}` }), t('delete'), t('cancel'))) {
            onClose?.()
            onChange((p) => roster.removeStudents(p, [index]))
          }
        }}
      />
    </View>
  )
}

function ValueField({ column, value, onCommit, onAddOption, flagMark }: { column: ColumnSpec; value: string; onCommit: (v: string) => void; onAddOption: (v: string) => void; flagMark: string }) {
  const { t } = useI18n()
  const [text, setText] = useState(value)
  const [lastValue, setLastValue] = useState(value)
  const [adding, setAdding] = useState(false)
  const [option, setOption] = useState('')
  // 名簿側の値が変わったら手元の値も合わせる
  if (value !== lastValue) {
    setLastValue(value)
    setText(value)
  }
  const commit = (v: string) => {
    const t = v.trim()
    if (t !== value) onCommit(t)
  }
  const addOption = () => {
    const v = option.trim()
    if (v) {
      // 足した選択肢をそのまま選ぶ（既にあれば選ぶだけ）
      if (!column.levels.includes(v)) onAddOption(v)
      onCommit(v)
    }
    setOption('')
    setAdding(false)
  }
  const label = (
    <Text style={styles.label}>
      {column.name} <Text style={{ color: C.muted, fontWeight: '600' }}>({t(KIND_KEY[column.kind])})</Text>
    </Text>
  )
  if (column.kind === 'flag') {
    const checked = value !== ''
    return (
      <Pressable
        accessibilityRole="checkbox"
        accessibilityState={{ checked }}
        accessibilityLabel={column.name}
        onPress={() => onCommit(checked ? '' : (column.levels[0] ?? flagMark))}
        style={(st: { pressed: boolean; hovered?: boolean }) => [
          { flexDirection: 'row', alignItems: 'center', gap: 10, minHeight: 44, paddingHorizontal: 10, borderRadius: 12, borderWidth: 1, borderColor: checked ? C.primary : C.border },
          { backgroundColor: checked ? C.primarySoft : st.pressed || st.hovered ? C.hover : C.card },
        ]}
      >
        <Ionicons name={checked ? 'checkbox' : 'square-outline'} size={24} color={checked ? C.primary : C.muted} />
        <Text style={{ flex: 1, fontSize: 15, fontWeight: '700', color: C.text }}>{column.name}</Text>
      </Pressable>
    )
  }
  if (column.kind === 'degree')
    return (
      <View>
        {label}
        <View style={styles.wrap}>
          {DEGREE_LEVELS.map((l) => (
            <Chip key={l} label={l} selected={value === l} onPress={() => onCommit(value === l ? '' : l)} />
          ))}
        </View>
      </View>
    )
  if (column.kind === 'category')
    return (
      <View>
        {label}
        {/* 単一選択: 選んでいるものをもう一度押すと空欄に戻る */}
        <View style={styles.wrap}>
          {column.levels.map((l) => (
            <Chip key={l} label={l} selected={value === l} onPress={() => onCommit(value === l ? '' : l)} />
          ))}
          {adding ? null : <Chip label="＋" accessibilityLabel={t('addOptionChip')} onPress={() => setAdding(true)} />}
        </View>
        {adding ? (
          <View style={[styles.row, { marginTop: 8 }]}>
            <TextInput
              style={[styles.input, { flex: 1 }]}
              value={option}
              onChangeText={setOption}
              onSubmitEditing={addOption}
              autoFocus
              returnKeyType="done"
              placeholder={t('newValuePlaceholder')}
              placeholderTextColor={C.muted}
              accessibilityLabel={t('valueA11y', { col: column.name })}
            />
            <Btn small variant="soft" icon="add" accessibilityLabel={t('add')} disabled={!option.trim()} onPress={addOption} />
            <Btn small icon="close" accessibilityLabel={t('cancel')} onPress={() => (setOption(''), setAdding(false))} />
          </View>
        ) : null}
      </View>
    )
  return (
    <View>
      {label}
      <TextInput
        style={styles.input}
        value={text}
        onChangeText={setText}
        onBlur={() => commit(text)}
        onSubmitEditing={() => commit(text)}
        keyboardType="decimal-pad"
        placeholder={t('numberPlaceholder')}
        accessibilityLabel={t('valueA11y', { col: column.name })}
      />
    </View>
  )
}
