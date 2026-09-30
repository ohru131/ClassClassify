import { useRef, useState } from 'react'
import { Text, TextInput, View } from 'react-native'

import { confirmAction } from '@/lib/confirm'
import { useI18n } from '@/lib/language-provider'
import { roster, type ColumnSpec, type Problem } from '@/lib/solver'
import { C } from './theme'
import { Btn, Chip, styles } from './ui'

const KIND_KEY = { flag: 'kindFlag', category: 'kindCategory', numeric: 'kindNumeric' } as const

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
        <ValueField key={c.name} column={c} value={s.values[c.name] ?? ''} onCommit={(v) => setValue(c.name, v)} flagMark={lang === 'ja' ? '○' : '✓'} />
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

function ValueField({ column, value, onCommit, flagMark }: { column: ColumnSpec; value: string; onCommit: (v: string) => void; flagMark: string }) {
  const { t } = useI18n()
  // カテゴリは選択肢のチップで選べるので、入力欄には選択肢に無い値だけを出す
  const initial = column.kind === 'category' && column.levels.includes(value) ? '' : value
  const [text, setText] = useState(initial)
  const [lastValue, setLastValue] = useState(value)
  // 名簿側の値が変わったら（チップで選んだときなど）手元の値も合わせる
  if (value !== lastValue) {
    setLastValue(value)
    setText(initial)
  }
  const commit = (v: string) => {
    const t = v.trim()
    // カテゴリの入力欄が空のまま確定しても、チップで選んだ値は消さない
    if (column.kind === 'category' && t === '') return
    if (t !== value) onCommit(t)
  }
  return (
    <View>
      <Text style={styles.label}>
        {column.name} <Text style={{ color: C.muted, fontWeight: '600' }}>({t(KIND_KEY[column.kind])})</Text>
      </Text>
      {column.kind === 'flag' ? (
        <View style={styles.wrap}>
          <Chip label={column.levels[0] ?? flagMark} selected={value !== ''} onPress={() => onCommit(value !== '' ? '' : (column.levels[0] ?? flagMark))} />
          <Chip label={t('blank')} selected={value === ''} onPress={() => onCommit('')} />
        </View>
      ) : column.kind === 'category' ? (
        <View style={{ gap: 8 }}>
          <View style={styles.wrap}>
            {column.levels.map((l) => (
              <Chip key={l} label={l} selected={value === l} onPress={() => onCommit(l)} />
            ))}
            <Chip label={t('blank')} selected={value === ''} onPress={() => onCommit('')} />
          </View>
          <TextInput
            style={styles.input}
            value={text}
            onChangeText={setText}
            onBlur={() => commit(text)}
            onSubmitEditing={() => commit(text)}
            placeholder={t('newValuePlaceholder')}
            accessibilityLabel={t('valueA11y', { col: column.name })}
          />
        </View>
      ) : (
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
      )}
    </View>
  )
}
