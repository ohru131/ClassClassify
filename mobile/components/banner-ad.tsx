import { memo } from 'react'
import { View } from 'react-native'
import { BannerAd, BannerAdSize, TestIds } from 'react-native-google-mobile-ads'

import { useAds } from '@/lib/ads-provider'

// 本番のユニット ID が未設定なら Google のテスト広告ユニット ID を使う
const BANNER_UNIT_ID = process.env.EXPO_PUBLIC_ADMOB_BANNER_UNIT_ID || TestIds.BANNER
export const BANNER_HEIGHT = 50

/** 無料版にだけ出すバナー。Pro・Web・同意前は何も描かない */
export const BannerAdSlot = memo(function BannerAdSlot() {
  const { isBannerVisible } = useAds()
  if (!isBannerVisible) return null
  return (
    // 読み込み完了で画面が跳ねないよう高さを確保する
    <View style={{ alignItems: 'center', minHeight: BANNER_HEIGHT }}>
      <BannerAd unitId={BANNER_UNIT_ID} size={BannerAdSize.BANNER} />
    </View>
  )
})
