import Ionicons from '@expo/vector-icons/Ionicons'
import { useState } from 'react'
import { Modal, Pressable, Text, View } from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'

import { useI18n } from '@/lib/language-provider'
import { useLayout } from '@/lib/layout'
import { C } from './theme'
import { Btn, type IconName } from './ui'

export type ExportChoice = { icon: IconName; label: string; sub: string; onPress: () => void }

/**
 * 形式ごとの書き出しボタン（Excel・PDF）。送り先（共有する・ファイルに保存）は押したあとに選ぶ。
 * 送り先が1つしか無い（iOS・Web）か Pro でない（押すと Pro の画面へ）ときは、選ばせずにそのまま動く。
 */
export function ExportButton({
  label,
  icon,
  title,
  choices,
  isPro,
  busy,
  variant,
}: {
  label: string
  icon: IconName
  /** 選択画面の見出し */
  title: string
  choices: ExportChoice[]
  isPro: boolean
  busy?: boolean
  variant?: 'primary' | 'ghost' | 'soft'
}) {
  const { t } = useI18n()
  const { isWide } = useLayout()
  const [open, setOpen] = useState(false)
  // Modal は端から端まで描かれる（edge-to-edge）。下のナビゲーションバーにキャンセルが隠れないようにする
  const insets = useSafeAreaInsets()
  const pick = (c: ExportChoice) => {
    setOpen(false)
    c.onPress()
  }
  return (
    <>
      <Btn
        small
        variant={variant}
        icon={icon}
        label={label}
        // Pro でないときは押すと Pro の画面へ移ることを読み上げでも伝える
        accessibilityLabel={isPro ? undefined : `${label} (Pro)`}
        busy={busy}
        onPress={() => (!isPro || choices.length === 1 ? choices[0].onPress() : setOpen(true))}
      />
      <Modal visible={open} transparent animationType="fade" onRequestClose={() => setOpen(false)}>
        {/* 背景を押しても閉じる。読み上げでは画面全体のボタンにせず、下のキャンセルと戻る操作で閉じる */}
        <Pressable
          accessible={false}
          importantForAccessibility="no"
          onPress={() => setOpen(false)}
          style={{ flex: 1, backgroundColor: 'rgba(15,23,42,0.4)', justifyContent: isWide ? 'center' : 'flex-end', alignItems: 'center', padding: isWide ? 24 : 0 }}
        >
          {/* 中を押しても閉じないよう、ここで押下を受け止める */}
          <Pressable
            onPress={() => {}}
            accessible={false}
            style={{
              width: '100%',
              maxWidth: 440,
              backgroundColor: C.card,
              borderRadius: 18,
              borderBottomLeftRadius: isWide ? 18 : 0,
              borderBottomRightRadius: isWide ? 18 : 0,
              padding: 16,
              paddingBottom: isWide ? 16 : 16 + insets.bottom,
              gap: 10,
            }}
          >
            <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }} accessibilityRole="header">
              {title}
            </Text>
            {choices.map((c) => (
              <Pressable
                key={c.label}
                accessibilityRole="button"
                accessibilityLabel={`${c.label}. ${c.sub}`}
                onPress={() => pick(c)}
                style={({ pressed }) => ({
                  flexDirection: 'row',
                  alignItems: 'center',
                  gap: 12,
                  padding: 14,
                  minHeight: 56,
                  borderRadius: 14,
                  borderWidth: 1,
                  borderColor: C.border,
                  backgroundColor: pressed ? C.hover : C.card,
                })}
              >
                <Ionicons name={c.icon} size={22} color={C.primary} />
                <View style={{ flex: 1 }}>
                  <Text style={{ fontSize: 15, fontWeight: '700', color: C.text }}>{c.label}</Text>
                  <Text style={{ fontSize: 12, color: C.sub, marginTop: 2 }}>{c.sub}</Text>
                </View>
              </Pressable>
            ))}
            <Btn label={t('cancel')} onPress={() => setOpen(false)} />
          </Pressable>
        </Pressable>
      </Modal>
    </>
  )
}

/** Pro 限定のまとまりに付ける小さな印 */
export function ProBadge() {
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 3, paddingHorizontal: 8, paddingVertical: 2, borderRadius: 999, backgroundColor: C.primarySoft }}>
      <Ionicons name="star" size={11} color={C.primaryText} />
      <Text style={{ fontSize: 11, fontWeight: '800', color: C.primaryText }}>Pro</Text>
    </View>
  )
}
