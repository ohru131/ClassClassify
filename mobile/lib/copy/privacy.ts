import type { AppLanguage } from '../i18n'

// プライバシーポリシー本文。実態と食い違わないこと（外部へ送るのは購入確認のため RevenueCat が
// 受け取る匿名の識別子とレシートだけ）。変えたら6言語とも、ストアのデータセーフティの申告も直す。
export type PrivacySection = { title: string; body: string[] }

const RC = 'https://www.revenuecat.com/privacy'
const GH = 'https://github.com/ohru131/ClassClassify'

export const PRIVACY: Record<AppLanguage, PrivacySection[]> = {
  ja: [
    {
      title: '名簿・編成結果',
      body: [
        '読み込んだ名簿（生徒の名前・特性・ペア指定）と編成結果は、この端末の中（アプリの保存領域）だけに保存します。開発者のサーバーやその他の外部へ送信することはありません。',
        '名簿の最適化（クラス編成の計算）も端末の中で行います。',
        'Excel ファイルや PDF として書き出した場合、そのファイルを送る先（Excel、Google ドライブ、メールなど）は OS の共有画面で利用者が選びます。送信先での取り扱いは各サービスの規約に従います。',
        '「Pro・設定」タブの「この端末の名簿と結果を消去」、またはアプリの削除で、保存したデータを消去できます。',
      ],
    },
    {
      title: '購入（Pro）と外部への送信',
      body: [
        '広告は表示しません（広告 SDK を組み込んでいません）。',
        'Pro は買い切り（非消費型）の購入です。決済は Apple App Store または Google Play が行い、購入状態の管理に RevenueCat を利用しています。',
        'RevenueCat は、購入を確認するために端末で生成された匿名の識別子と、購入・レシートの情報を受け取ります。このアプリから名前やメールアドレス、名簿の内容を送ることはありません。購入の確認以外の目的（広告・行動分析など）で外部へ情報を送ることもありません。',
        `RevenueCat のプライバシーポリシー: ${RC}`,
      ],
    },
    { title: 'お問い合わせ', body: [`ご質問は GitHub（${GH}）の Issues からお寄せください。`] },
  ],
  en: [
    {
      title: 'Rosters and results',
      body: [
        'The rosters you load (student names, attributes and pairings) and the class placements are stored only on this device, in the app’s own storage. They are never sent to the developer’s servers or anywhere else.',
        'The optimization (building the classes) also runs entirely on this device.',
        'When you export an Excel file or PDF, you choose where it goes (Excel, Google Drive, email, etc.) in your device’s share sheet. That service’s own terms then apply.',
        'You can erase the saved data with “Erase roster and results on this device” in the Pro & settings tab, or by uninstalling the app.',
      ],
    },
    {
      title: 'Purchases (Pro) and what is sent',
      body: [
        'The app shows no ads and contains no advertising SDK.',
        'Pro is a one-time (non-consumable) purchase. Payment is handled by the Apple App Store or Google Play, and RevenueCat is used to manage purchase status.',
        'To verify your purchase, RevenueCat receives an anonymous identifier generated on your device and the purchase/receipt information. The app never sends your name, email address or any roster data, and sends nothing for any other purpose (such as advertising or analytics).',
        `RevenueCat privacy policy: ${RC}`,
      ],
    },
    { title: 'Contact', body: [`Please send questions via Issues on GitHub (${GH}).`] },
  ],
  ko: [
    {
      title: '명단과 편성 결과',
      body: [
        '불러온 명단(학생 이름, 특성, 배정 조건)과 편성 결과는 이 기기(앱 저장 공간)에만 저장합니다. 개발자 서버나 그 밖의 외부로 전송하지 않습니다.',
        '반 편성 계산도 이 기기 안에서 합니다.',
        '엑셀 파일이나 PDF로 내보낼 때는 보낼 곳(엑셀, Google 드라이브, 메일 등)을 기기의 공유 화면에서 직접 선택합니다. 보낸 곳에서의 처리는 해당 서비스의 약관을 따릅니다.',
        '"Pro·설정" 탭의 "이 기기의 명단과 결과 지우기" 또는 앱 삭제로 저장된 데이터를 지울 수 있습니다.',
      ],
    },
    {
      title: '구매(Pro)와 외부 전송',
      body: [
        '광고를 표시하지 않으며 광고 SDK도 포함하지 않습니다.',
        'Pro는 1회 구매(비소모성) 상품입니다. 결제는 Apple App Store 또는 Google Play가 처리하고, 구매 상태 관리에 RevenueCat을 사용합니다.',
        'RevenueCat은 구매를 확인하기 위해 기기에서 생성한 익명 식별자와 구매·영수증 정보를 받습니다. 앱은 이름, 이메일 주소, 명단 내용을 보내지 않으며, 구매 확인 외의 목적(광고, 행동 분석 등)으로 정보를 보내지 않습니다.',
        `RevenueCat 개인정보 처리방침: ${RC}`,
      ],
    },
    { title: '문의', body: [`문의는 GitHub(${GH})의 Issues로 보내 주세요.`] },
  ],
  es: [
    {
      title: 'Listas y resultados',
      body: [
        'Las listas que cargas (nombres de los estudiantes, criterios y condiciones) y la distribución en grupos se guardan solo en este dispositivo, en el almacenamiento de la app. Nunca se envían a los servidores del desarrollador ni a ningún otro lugar.',
        'La optimización (armar los grupos) también se hace completamente en este dispositivo.',
        'Cuando exportas un Excel o un PDF, tú eliges a dónde enviarlo (Excel, Google Drive, correo, etc.) desde el menú de compartir del dispositivo. A partir de ahí se aplican las condiciones de ese servicio.',
        'Puedes borrar los datos guardados con «Borrar la lista y los resultados de este dispositivo» en la pestaña Pro y ajustes, o desinstalando la app.',
      ],
    },
    {
      title: 'Compras (Pro) y lo que se envía',
      body: [
        'La app no muestra anuncios ni incluye ningún SDK de publicidad.',
        'Pro es una compra de pago único (no consumible). El pago lo gestiona Apple App Store o Google Play, y se usa RevenueCat para administrar el estado de la compra.',
        'Para verificar la compra, RevenueCat recibe un identificador anónimo generado en tu dispositivo y la información de compra y recibo. La app nunca envía tu nombre, tu correo ni los datos de la lista, ni envía información con otros fines (como publicidad o análisis).',
        `Política de privacidad de RevenueCat: ${RC}`,
      ],
    },
    { title: 'Contacto', body: [`Envía tus preguntas en Issues de GitHub (${GH}).`] },
  ],
  de: [
    {
      title: 'Schülerlisten und Ergebnisse',
      body: [
        'Die geladenen Schülerlisten (Namen, Merkmale und Wünsche) und die Klasseneinteilung werden nur auf diesem Gerät im Speicher der App abgelegt. Sie werden weder an Server des Entwicklers noch an Dritte übertragen.',
        'Auch die Berechnung der Einteilung findet vollständig auf diesem Gerät statt.',
        'Beim Export als Excel-Datei oder PDF wählen Sie im Teilen-Menü des Geräts selbst, wohin die Datei geht (Excel, Google Drive, E-Mail usw.). Dort gelten die Bedingungen des jeweiligen Dienstes.',
        'Gespeicherte Daten löschen Sie über „Liste und Ergebnis auf diesem Gerät löschen“ im Tab Pro oder durch Deinstallieren der App.',
      ],
    },
    {
      title: 'Kauf (Pro) und übertragene Daten',
      body: [
        'Die App zeigt keine Werbung und enthält kein Werbe-SDK.',
        'Pro ist ein Einmalkauf (nicht verbrauchbar). Die Zahlung wickelt der Apple App Store oder Google Play ab; zur Verwaltung des Kaufstatus wird RevenueCat genutzt.',
        'Zur Prüfung des Kaufs erhält RevenueCat eine auf Ihrem Gerät erzeugte anonyme Kennung sowie die Kauf- bzw. Belegdaten. Die App überträgt weder Ihren Namen noch Ihre E-Mail-Adresse noch Inhalte der Schülerliste und sendet keine Daten zu anderen Zwecken (z. B. Werbung oder Analyse).',
        `Datenschutzerklärung von RevenueCat: ${RC}`,
      ],
    },
    { title: 'Kontakt', body: [`Fragen bitte über die Issues auf GitHub (${GH}).`] },
  ],
  'pt-BR': [
    {
      title: 'Listas e resultados',
      body: [
        'As listas que você carrega (nomes dos alunos, critérios e condições) e a enturmação ficam salvas só neste aparelho, no armazenamento do app. Nunca são enviadas aos servidores do desenvolvedor nem a qualquer outro lugar.',
        'A otimização (montar as turmas) também é feita inteiramente neste aparelho.',
        'Ao exportar um Excel ou PDF, você escolhe para onde enviar (Excel, Google Drive, e-mail etc.) no menu de compartilhar do aparelho. A partir daí valem os termos desse serviço.',
        'Você pode apagar os dados salvos em «Apagar a lista e o resultado deste aparelho», na aba Pro e ajustes, ou desinstalando o app.',
      ],
    },
    {
      title: 'Compras (Pro) e o que é enviado',
      body: [
        'O app não exibe anúncios nem inclui SDK de publicidade.',
        'O Pro é uma compra única (não consumível). O pagamento é processado pela Apple App Store ou pelo Google Play, e o RevenueCat é usado para gerenciar o status da compra.',
        'Para verificar a compra, o RevenueCat recebe um identificador anônimo gerado no seu aparelho e as informações da compra e do recibo. O app nunca envia seu nome, e-mail ou dados da lista, nem envia informações para outros fins (como publicidade ou análise).',
        `Política de privacidade do RevenueCat: ${RC}`,
      ],
    },
    { title: 'Contato', body: [`Envie dúvidas pelas Issues do GitHub (${GH}).`] },
  ],
}
