import Ionicons from '@expo/vector-icons/Ionicons'
import type { ComponentProps, ReactNode } from 'react'
import { ActivityIndicator, Platform, Pressable, ScrollView, StyleSheet, Text, View, type StyleProp, type ViewStyle } from 'react-native'

import { useLayout } from '@/lib/layout'
import { C } from './theme'

export type IconName = ComponentProps<typeof Ionicons>['name']

// Web（react-native-web）の Pressable は style 関数に hovered / focused も渡す。
// ネイティブでは pressed だけ。マウス・キーボード（Chromebook）でも押せる場所が分かるようにする。
type InteractionState = { pressed: boolean; hovered?: boolean; focused?: boolean }

const focusRing = (s: InteractionState): ViewStyle | null =>
  s.focused ? (Platform.OS === 'web' ? ({ outlineColor: C.focus, outlineWidth: 2, outlineStyle: 'solid', outlineOffset: 1 } as ViewStyle) : { borderColor: C.focus }) : null

export function Btn({
  label,
  icon,
  onPress,
  variant = 'ghost',
  disabled,
  busy,
  small,
  style,
  accessibilityLabel,
}: {
  label?: string
  icon?: IconName
  onPress?: () => void
  variant?: 'primary' | 'ghost' | 'danger' | 'soft'
  disabled?: boolean
  busy?: boolean
  small?: boolean
  style?: StyleProp<ViewStyle>
  accessibilityLabel?: string
}) {
  const fg = variant === 'primary' ? '#fff' : variant === 'danger' ? C.danger : variant === 'soft' ? C.primaryText : C.text
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel ?? label}
      accessibilityState={{ disabled: !!disabled, busy: !!busy }}
      disabled={disabled || busy}
      onPress={onPress}
      style={(st: InteractionState) => [
        styles.btn,
        small && styles.btnSmall,
        variant === 'primary' && { backgroundColor: st.pressed ? '#4338CA' : st.hovered ? '#4F46E5' : C.primary, borderColor: 'transparent' },
        variant === 'ghost' && { backgroundColor: st.pressed || st.hovered ? C.hover : C.card },
        variant === 'soft' && { backgroundColor: st.pressed || st.hovered ? '#E0E7FF' : C.primarySoft, borderColor: 'transparent' },
        variant === 'danger' && { backgroundColor: st.pressed || st.hovered ? '#FFE4E6' : C.dangerSoft, borderColor: 'transparent' },
        (disabled || busy) && { opacity: 0.45 },
        focusRing(st),
        style,
      ]}
    >
      {busy ? <ActivityIndicator size="small" color={fg} /> : icon ? <Ionicons name={icon} size={small ? 15 : 18} color={fg} /> : null}
      {label ? <Text style={[styles.btnText, small && { fontSize: 13 }, { color: fg }]}>{label}</Text> : null}
    </Pressable>
  )
}

export function Chip({
  label,
  selected,
  onPress,
  bg,
  fg,
  accessibilityLabel,
}: {
  label: string
  selected?: boolean
  onPress?: () => void
  bg?: string
  fg?: string
  accessibilityLabel?: string
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel ?? label}
      accessibilityState={{ selected: !!selected }}
      onPress={onPress}
      disabled={!onPress}
      style={(st: InteractionState) => [
        styles.chip,
        { backgroundColor: selected ? C.primary : (bg ?? (st.hovered || st.pressed ? C.hover : C.card)) },
        selected && { borderColor: C.primary },
        focusRing(st),
      ]}
    >
      <Text style={[styles.chipText, { color: selected ? '#fff' : (fg ?? C.sub) }]}>{label}</Text>
    </Pressable>
  )
}

export function Segmented<T extends string | number>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <View style={styles.seg} accessibilityRole="tablist">
      {options.map((o) => {
        const active = o.value === value
        return (
          <Pressable
            key={String(o.value)}
            accessibilityRole="tab"
            accessibilityState={{ selected: active }}
            onPress={() => onChange(o.value)}
            style={(st: InteractionState) => [styles.segItem, active ? styles.segActive : (st.hovered || st.pressed) && { backgroundColor: '#E2E8F0' }, focusRing(st)]}
          >
            <Text style={[styles.segText, active && { color: C.text }]} numberOfLines={1}>
              {o.label}
            </Text>
          </Pressable>
        )
      })}
    </View>
  )
}

export function Stepper({ value, onChange, min, max, step = 1, format, label }: { value: number; onChange: (v: number) => void; min: number; max: number; step?: number; format?: (v: number) => string; label: string }) {
  return (
    <View style={styles.row}>
      <Btn icon="remove" small accessibilityLabel={`${label}を減らす`} disabled={value <= min} onPress={() => onChange(Math.max(min, Math.round((value - step) * 100) / 100))} />
      <Text style={styles.stepValue} accessibilityLabel={`${label} ${value}`}>
        {format ? format(value) : value}
      </Text>
      <Btn icon="add" small accessibilityLabel={`${label}を増やす`} disabled={value >= max} onPress={() => onChange(Math.min(max, Math.round((value + step) * 100) / 100))} />
    </View>
  )
}

export function Card({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[styles.card, style]}>{children}</View>
}

export function Title({ children, sub }: { children: ReactNode; sub?: string }) {
  return (
    <View style={{ marginBottom: 10 }}>
      <Text style={styles.title} accessibilityRole="header">
        {children}
      </Text>
      {sub ? <Text style={styles.sub}>{sub}</Text> : null}
    </View>
  )
}

export function Stat({ label, value, sub, tone = 'default' }: { label: string; value: string | number; sub: string; tone?: 'default' | 'good' | 'bad' }) {
  const color = tone === 'good' ? C.good : tone === 'bad' ? C.danger : C.text
  return (
    <View style={[styles.card, styles.stat]}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={[styles.statValue, { color }]}>{value}</Text>
      <Text style={styles.statSub}>{sub}</Text>
    </View>
  )
}

export function Notice({ tone, children, onClose }: { tone: 'error' | 'warn' | 'good' | 'info'; children: ReactNode; onClose?: () => void }) {
  const palette = { error: [C.dangerSoft, C.danger], warn: [C.warnSoft, C.warn], good: [C.goodSoft, C.good], info: [C.primarySoft, C.primaryText] }[tone]
  return (
    <View style={[styles.notice, { backgroundColor: palette[0] }]} accessibilityLiveRegion="polite">
      <View style={{ flex: 1 }}>{typeof children === 'string' ? <Text style={{ color: palette[1], fontSize: 14 }}>{children}</Text> : children}</View>
      {onClose ? <Btn icon="close" small variant="ghost" accessibilityLabel="閉じる" onPress={onClose} style={{ backgroundColor: 'transparent', borderWidth: 0 }} /> : null}
    </View>
  )
}

/** 各タブの本文。大画面では最大幅で中央に寄せる */
export function Screen({ children, scroll = true }: { children: ReactNode; scroll?: boolean }) {
  const { contentMaxWidth, isWide } = useLayout()
  const inner = <View style={[styles.screenInner, { maxWidth: contentMaxWidth, padding: isWide ? 24 : 14 }]}>{children}</View>
  if (!scroll) return <View style={{ flex: 1, backgroundColor: C.bg }}>{inner}</View>
  return (
    <ScrollView style={{ flex: 1, backgroundColor: C.bg }} contentContainerStyle={{ paddingBottom: 40 }} keyboardShouldPersistTaps="handled">
      {inner}
    </ScrollView>
  )
}

export const styles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  wrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    minHeight: 44,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: C.border,
  },
  btnSmall: { minHeight: 36, paddingHorizontal: 10, borderRadius: 10 },
  btnText: { fontSize: 15, fontWeight: '700' },
  chip: { paddingHorizontal: 12, minHeight: 34, justifyContent: 'center', borderRadius: 999, borderWidth: 1, borderColor: C.border },
  chipText: { fontSize: 13, fontWeight: '700' },
  seg: { flexDirection: 'row', backgroundColor: '#EEF0F5', borderRadius: 12, padding: 3, gap: 2 },
  segItem: { flex: 1, minHeight: 38, paddingHorizontal: 8, alignItems: 'center', justifyContent: 'center', borderRadius: 10 },
  segActive: { backgroundColor: '#fff', shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 3, shadowOffset: { width: 0, height: 1 }, elevation: 1 },
  segText: { fontSize: 13, fontWeight: '700', color: C.sub },
  stepValue: { minWidth: 44, textAlign: 'center', fontSize: 22, fontWeight: '800', color: C.text, fontVariant: ['tabular-nums'] },
  card: { backgroundColor: C.card, borderRadius: 18, borderWidth: 1, borderColor: C.border, padding: 16 },
  title: { fontSize: 20, fontWeight: '800', color: C.text },
  sub: { fontSize: 13, color: C.sub, marginTop: 4, lineHeight: 19 },
  stat: { flexGrow: 1, flexBasis: 150, padding: 14 },
  statLabel: { fontSize: 12, fontWeight: '700', color: C.sub },
  statValue: { fontSize: 26, fontWeight: '800', marginTop: 2 },
  statSub: { fontSize: 11, color: C.muted, marginTop: 2 },
  notice: { flexDirection: 'row', alignItems: 'center', borderRadius: 14, paddingHorizontal: 14, paddingVertical: 10, gap: 8 },
  screenInner: { width: '100%', alignSelf: 'center', gap: 14 },
  input: {
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 10,
    paddingHorizontal: 12,
    minHeight: 44,
    fontSize: 16,
    color: C.text,
    backgroundColor: '#fff',
  },
  label: { fontSize: 12, fontWeight: '700', color: C.sub, marginBottom: 4 },
})
