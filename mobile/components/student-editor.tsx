import { useRef, useState } from 'react'
import { Text, TextInput, View } from 'react-native'

import { confirmAction } from '@/lib/confirm'
import { roster, type ColumnSpec, type Problem } from '@/lib/solver'
import { C } from './theme'
import { Btn, Chip, styles } from './ui'

const KIND_LABEL: Record<ColumnSpec['kind'], string> = { flag: '該当', category: 'カテゴリ', numeric: '数値' }

/**
 * 生徒1人の編集。テキストは入力中は手元に持ち、確定（Enter・フォーカスが外れたとき）で名簿へ反映する
 * （1文字ごとに項目の種類を判定し直すと、入力の途中で欄の種類が変わってしまうため）。
 * 呼び出し側で key に生徒の index を渡し、別の生徒に切り替えたら作り直すこと。
 */
export function StudentEditor({ problem, index, onChange, onClose }: { problem: Problem; index: number; onChange: (f: (p: Problem) => Problem) => void; onClose?: () => void }) {
  const s = problem.students[index]
  const [no, setNo] = useState(String(s.no))
  const [name, setName] = useState(s.name)
  const [noError, setNoError] = useState<string | null>(null)
  const nameRef = useRef<TextInput>(null)
  const groups = roster.groupsOf(problem)

  const commitNo = () => {
    const v = Number(no)
    if (!Number.isInteger(v) || v <= 0) return setNoError('1以上の整数を入力してください')
    if (roster.isNoTaken(problem, v, index)) return setNoError(`NO ${v} は他の生徒が使っています`)
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
        <Text style={{ fontSize: 18, fontWeight: '800', color: C.text }}>生徒の編集</Text>
        {onClose ? <Btn small icon="close" accessibilityLabel="閉じる" onPress={onClose} /> : null}
      </View>
      <View style={styles.row}>
        <View style={{ width: 96 }}>
          <Text style={styles.label}>NO</Text>
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
            accessibilityLabel="出席番号"
          />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.label}>名前</Text>
          <TextInput
            ref={nameRef}
            style={styles.input}
            value={name}
            onChangeText={setName}
            onBlur={commitName}
            onSubmitEditing={commitName}
            returnKeyType="done"
            placeholder="氏名"
            accessibilityLabel="名前"
          />
        </View>
      </View>
      {noError ? <Text style={{ color: C.danger, fontSize: 12 }}>{noError}</Text> : null}

      {problem.columns.map((c) => (
        <ValueField key={c.name} column={c} value={s.values[c.name] ?? ''} onCommit={(v) => setValue(c.name, v)} />
      ))}
      {problem.columns.length === 0 ? <Text style={{ color: C.muted }}>項目がありません。「項目」タブから追加できます。</Text> : null}

      {groups.wanted[index].length + groups.unwanted[index].length > 0 ? (
        <View>
          <Text style={styles.label}>ペア指定</Text>
          <View style={styles.wrap}>
            {groups.wanted[index].map((g) => (
              <Chip key={`w${g}`} label={`同${g + 1}: ${problem.wantedGroups[g].map((i) => problem.students[i].name || problem.students[i].no).join('・')}`} bg="#E0F2FE" fg="#0369A1" />
            ))}
            {groups.unwanted[index].map((g) => (
              <Chip key={`u${g}`} label={`別${g + 1}: ${problem.unwantedGroups[g].map((i) => problem.students[i].name || problem.students[i].no).join('・')}`} bg="#FFE4E6" fg="#BE123C" />
            ))}
          </View>
        </View>
      ) : null}

      <Btn
        variant="danger"
        icon="trash-outline"
        label="この生徒を削除"
        onPress={async () => {
          if (await confirmAction('生徒を削除', `${s.no} ${s.name || '（名前なし）'} を名簿から削除します。ペア指定からも外れます。`, '削除')) {
            onClose?.()
            onChange((p) => roster.removeStudents(p, [index]))
          }
        }}
      />
    </View>
  )
}

function ValueField({ column, value, onCommit }: { column: ColumnSpec; value: string; onCommit: (v: string) => void }) {
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
        {column.name} <Text style={{ color: C.muted, fontWeight: '600' }}>（{KIND_LABEL[column.kind]}）</Text>
      </Text>
      {column.kind === 'flag' ? (
        <View style={styles.wrap}>
          <Chip label={column.levels[0] ?? '○'} selected={value !== ''} onPress={() => onCommit(value !== '' ? '' : (column.levels[0] ?? '○'))} />
          <Chip label="空欄" selected={value === ''} onPress={() => onCommit('')} />
        </View>
      ) : column.kind === 'category' ? (
        <View style={{ gap: 8 }}>
          <View style={styles.wrap}>
            {column.levels.map((l) => (
              <Chip key={l} label={l} selected={value === l} onPress={() => onCommit(l)} />
            ))}
            <Chip label="空欄" selected={value === ''} onPress={() => onCommit('')} />
          </View>
          <TextInput
            style={styles.input}
            value={text}
            onChangeText={setText}
            onBlur={() => commit(text)}
            onSubmitEditing={() => commit(text)}
            placeholder="新しい値を入力して確定"
            accessibilityLabel={`${column.name}の値`}
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
          placeholder="数値"
          accessibilityLabel={`${column.name}の値`}
        />
      )}
    </View>
  )
}
