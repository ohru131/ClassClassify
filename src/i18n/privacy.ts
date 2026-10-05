import type { AppLanguage } from './languages'

// プライバシーポリシー本文。実態と食い違わないこと（アプリ自体は何も外部へ送らない。Pro の決済は
// Google Play が行い、購入の状態は端末の Play ストアに問い合わせるだけ）。変えたら6言語とも、ストアのデータセーフティの申告も直す。
export type PrivacySection = { title: string; body: string[] }

const GOOGLE_PRIVACY = 'https://policies.google.com/privacy'
const ISSUES_URL = 'https://github.com/ohru131/ClassClassify/issues'
const DEVELOPER = 'Katahimo'

/** 最終更新日（本文を変えたら更新する）。Web で公開しているページ（privacy/index.html）にも出す */
export const PRIVACY_UPDATED = '2026-10-05'

/**
 * 公開ページ（https://ohru131.github.io/ClassClassify/privacy/）の冒頭に出す「このポリシーの対象」。
 * アプリ内の画面では出さない（アプリの中にいる時点で対象は自明なため）。
 */
export const PRIVACY_SCOPE: Record<AppLanguage, string> = {
  ja: 'このプライバシーポリシーは、クラス編成アプリ「FairClass」のスマホ・タブレット版（Android / iOS）に適用されます。アプリ内の「Pro・設定」タブから開ける画面と同じ内容です。',
  en: 'This privacy policy applies to the FairClass class placement app for phones and tablets (Android / iOS). It is the same text as the screen you can open from the Pro tab in the app.',
  ko: '이 개인정보 처리방침은 반 편성 앱 "FairClass"의 스마트폰·태블릿 버전(Android / iOS)에 적용됩니다. 앱의 "Pro·설정" 탭에서 열 수 있는 화면과 같은 내용입니다.',
  es: 'Esta política de privacidad se aplica a la app FairClass para armar grupos y cursos en teléfonos y tabletas (Android / iOS). Es el mismo texto que la pantalla que se abre desde la pestaña Pro de la app.',
  de: 'Diese Datenschutzerklärung gilt für die App „FairClass“ zur Klasseneinteilung auf Smartphones und Tablets (Android / iOS). Sie entspricht dem Bildschirm, den Sie in der App im Tab Pro öffnen können.',
  'pt-BR': 'Esta política de privacidade se aplica ao app FairClass de enturmação para celulares e tablets (Android / iOS). É o mesmo texto da tela que se abre pela aba Pro do app.',
}

/** 公開ページの見出しまわり（「最終更新日」の言い方） */
export const PRIVACY_UPDATED_LABEL: Record<AppLanguage, string> = {
  ja: '最終更新日',
  en: 'Last updated',
  ko: '최종 수정일',
  es: 'Última actualización',
  de: 'Zuletzt aktualisiert',
  'pt-BR': 'Última atualização',
}

export const PRIVACY: Record<AppLanguage, PrivacySection[]> = {
  ja: [
    {
      title: '名簿・編成結果',
      body: [
        '読み込んだ名簿（生徒の名前・特性・ペア指定）と編成結果は、この端末の中（アプリの保存領域）だけに保存します。開発者のサーバーやその他の外部へ送信することはありません。',
        'クラス編成の組み合わせを考える作業も、端末の中で行います。',
        'Android 版はアプリのバックアップを無効にしているため、名簿は Google ドライブへの自動バックアップにも含まれません。iPhone・iPad では、利用者が管理する端末自体のバックアップ（iCloud バックアップやパソコンへのバックアップ）に含まれる場合があります。',
        'Excel ファイルや PDF として書き出した場合、そのファイルを送る先（Excel、Google ドライブ、メールなど）は OS の共有画面で利用者が選びます。送信先での取り扱いは各サービスの規約に従います。',
        '「Pro・設定」タブの「この端末の名簿と結果を消去」、またはアプリの削除で、保存したデータを消去できます。',
      ],
    },
    {
      title: '購入（Pro）と外部への送信',
      body: [
        '広告は表示しません（広告 SDK を組み込んでいません）。',
        'Pro は買い切り（非消費型）の購入です。決済は Google Play が行い、支払いの情報は Google のプライバシーポリシーに従って扱われます。',
        '購入済みかどうかは、端末の Play ストアに問い合わせて確認します。このアプリ自体は、購入の情報や識別子も含め、開発者のサーバーやその他の外部へ何も送信しません。名前やメールアドレス、名簿の内容を送ることも、広告・行動分析などの目的で情報を送ることもありません。',
        `Google のプライバシーポリシー: ${GOOGLE_PRIVACY}`,
      ],
    },
    {
      title: '開発者・お問い合わせ',
      body: [`開発者: ${DEVELOPER}`, `このポリシーやデータの扱いについてのお問い合わせは、GitHub の Issues へお寄せください: ${ISSUES_URL}`],
    },
  ],
  en: [
    {
      title: 'Rosters and results',
      body: [
        'The rosters you load (student names, attributes and pairings) and the class placements are stored only on this device, in the app’s own storage. They are never sent to the developer’s servers or anywhere else.',
        'Building the classes also happens entirely on this device.',
        'On Android, app backup is turned off, so rosters are not included in the automatic backup to Google Drive. On iPhone and iPad, they may be included in backups of the device itself that you manage (iCloud Backup or a backup to your computer).',
        'When you export an Excel file or PDF, you choose where it goes (Excel, Google Drive, email, etc.) in your device’s share sheet. That service’s own terms then apply.',
        'You can erase the saved data with “Erase roster and results on this device” in the Pro tab, or by uninstalling the app.',
      ],
    },
    {
      title: 'Purchases (Pro) and what is sent',
      body: [
        'The app shows no ads and contains no advertising SDK.',
        'Pro is a one-time (non-consumable) purchase. Payment is handled by Google Play, and your payment information is processed under Google’s privacy policy.',
        'Whether you own Pro is checked by asking the Play Store on your device. The app itself sends nothing to the developer’s servers or anywhere else, including purchase information or identifiers. It never sends your name, email address or any roster data, and sends nothing for advertising, analytics or any other purpose.',
        `Google privacy policy: ${GOOGLE_PRIVACY}`,
      ],
    },
    {
      title: 'Developer and contact',
      body: [`Developer: ${DEVELOPER}`, `For questions about this policy or how data is handled, please open an issue on GitHub: ${ISSUES_URL}`],
    },
  ],
  ko: [
    {
      title: '명단과 편성 결과',
      body: [
        '불러온 명단(학생 이름, 특성, 배정 조건)과 편성 결과는 이 기기(앱 저장 공간)에만 저장합니다. 개발자 서버나 그 밖의 외부로 전송하지 않습니다.',
        '반 편성 계산도 이 기기 안에서 합니다.',
        'Android 버전은 앱 백업을 사용하지 않으므로 명단이 Google 드라이브 자동 백업에도 포함되지 않습니다. iPhone·iPad에서는 사용자가 관리하는 기기 자체의 백업(iCloud 백업, 컴퓨터 백업)에 포함될 수 있습니다.',
        '엑셀 파일이나 PDF로 내보낼 때는 보낼 곳(엑셀, Google 드라이브, 메일 등)을 기기의 공유 화면에서 직접 선택합니다. 보낸 곳에서의 처리는 해당 서비스의 약관을 따릅니다.',
        '"Pro·설정" 탭의 "이 기기의 명단과 결과 지우기" 또는 앱 삭제로 저장된 데이터를 지울 수 있습니다.',
      ],
    },
    {
      title: '구매(Pro)와 외부 전송',
      body: [
        '광고를 표시하지 않으며 광고 SDK도 포함하지 않습니다.',
        'Pro는 1회 구매(비소모성) 상품입니다. 결제는 Google Play가 처리하며, 결제 정보는 Google 개인정보처리방침에 따라 처리됩니다.',
        '구매 여부는 기기의 Play 스토어에 조회해 확인합니다. 앱 자체는 구매 정보나 식별자를 포함해 어떤 정보도 개발자 서버나 그 밖의 외부로 보내지 않습니다. 이름, 이메일 주소, 명단 내용을 보내지 않으며, 광고·행동 분석 등의 목적으로 정보를 보내지도 않습니다.',
        `Google 개인정보처리방침: ${GOOGLE_PRIVACY}`,
      ],
    },
    {
      title: '개발자 및 문의',
      body: [`개발자: ${DEVELOPER}`, `이 방침이나 데이터 처리에 대한 문의는 GitHub Issues로 보내 주십시오: ${ISSUES_URL}`],
    },
  ],
  es: [
    {
      title: 'Listas y resultados',
      body: [
        'Las listas que cargas (nombres de los estudiantes, criterios y condiciones) y la distribución en grupos se guardan solo en este dispositivo, en el almacenamiento de la app. Nunca se envían a los servidores del desarrollador ni a ningún otro lugar.',
        'Armar los grupos también se hace por completo en este dispositivo.',
        'En Android, la copia de seguridad de la app está desactivada, así que las listas no se incluyen en la copia automática en Google Drive. En iPhone y iPad pueden incluirse en las copias de seguridad del propio dispositivo que tú administras (copia en iCloud o en tu computadora).',
        'Cuando exportas un Excel o un PDF, tú eliges a dónde enviarlo (Excel, Google Drive, correo, etc.) desde el menú de compartir del dispositivo. A partir de ahí se aplican las condiciones de ese servicio.',
        'Puedes borrar los datos guardados con “Borrar la lista y los resultados de este dispositivo” en la pestaña Pro, o desinstalando la app.',
      ],
    },
    {
      title: 'Compras (Pro) y lo que se envía',
      body: [
        'La app no muestra anuncios ni incluye ningún SDK de publicidad.',
        'Pro es una compra de pago único (no consumible). El pago lo gestiona Google Play, y tus datos de pago se tratan según la política de privacidad de Google.',
        'Para saber si ya tienes Pro, la app lo consulta a Play Store en tu dispositivo. La app en sí no envía nada a los servidores del desarrollador ni a ningún otro lugar, tampoco datos de compra ni identificadores. Nunca envía tu nombre, tu correo ni los datos de la lista, ni envía información con fines de publicidad, análisis u otros.',
        `Política de privacidad de Google: ${GOOGLE_PRIVACY}`,
      ],
    },
    {
      title: 'Desarrollador y contacto',
      body: [`Desarrollador: ${DEVELOPER}`, `Para consultas sobre esta política o el tratamiento de los datos, abre una incidencia en GitHub: ${ISSUES_URL}`],
    },
  ],
  de: [
    {
      title: 'Schülerlisten und Ergebnisse',
      body: [
        'Die geladenen Schülerlisten (Namen, Merkmale und Wünsche) und die Klasseneinteilung werden nur auf diesem Gerät im Speicher der App abgelegt. Sie werden weder an Server des Entwicklers noch an Dritte übertragen.',
        'Auch die Berechnung der Einteilung findet vollständig auf diesem Gerät statt.',
        'Unter Android ist die App-Sicherung deaktiviert; die Listen sind daher nicht in der automatischen Sicherung in Google Drive enthalten. Auf iPhone und iPad können sie in Sicherungen des Geräts enthalten sein, die Sie selbst verwalten (iCloud-Backup oder Backup auf dem Computer).',
        'Beim Export als Excel-Datei oder PDF wählen Sie im Teilen-Menü des Geräts selbst, wohin die Datei geht (Excel, Google Drive, E-Mail usw.). Dort gelten die Bedingungen des jeweiligen Dienstes.',
        'Gespeicherte Daten löschen Sie über „Liste und Ergebnis auf diesem Gerät löschen“ im Tab Pro oder durch Deinstallieren der App.',
      ],
    },
    {
      title: 'Kauf (Pro) und übertragene Daten',
      body: [
        'Die App zeigt keine Werbung und enthält kein Werbe-SDK.',
        'Pro ist ein Einmalkauf (nicht verbrauchbar). Die Zahlung wickelt Google Play ab; Ihre Zahlungsdaten werden gemäß der Datenschutzerklärung von Google verarbeitet.',
        'Ob Sie Pro bereits gekauft haben, fragt die App beim Play Store auf Ihrem Gerät ab. Die App selbst überträgt nichts an Server des Entwicklers oder an Dritte – auch keine Kaufdaten oder Kennungen. Sie überträgt weder Ihren Namen noch Ihre E-Mail-Adresse noch Inhalte der Schülerliste und sendet keine Daten zu Werbe-, Analyse- oder anderen Zwecken.',
        `Datenschutzerklärung von Google: ${GOOGLE_PRIVACY}`,
      ],
    },
    {
      title: 'Entwickler und Kontakt',
      body: [`Entwickler: ${DEVELOPER}`, `Fragen zu dieser Erklärung oder zum Umgang mit Daten richten Sie bitte als Issue auf GitHub an uns: ${ISSUES_URL}`],
    },
  ],
  'pt-BR': [
    {
      title: 'Listas e resultados',
      body: [
        'As listas que você carrega (nomes dos alunos, critérios e condições) e a enturmação ficam salvas só neste aparelho, no armazenamento do app. Nunca são enviadas aos servidores do desenvolvedor nem a qualquer outro lugar.',
        'A montagem das turmas também é feita inteiramente neste aparelho.',
        'No Android, o backup do app está desativado, então as listas não entram no backup automático do Google Drive. No iPhone e no iPad, elas podem entrar nos backups do próprio aparelho que você gerencia (backup do iCloud ou no computador).',
        'Ao exportar um Excel ou PDF, você escolhe para onde enviar (Excel, Google Drive, e-mail etc.) no menu de compartilhar do aparelho. A partir daí valem os termos desse serviço.',
        'Você pode apagar os dados salvos em “Apagar a lista e o resultado deste aparelho”, na aba Pro, ou desinstalando o app.',
      ],
    },
    {
      title: 'Compras (Pro) e o que é enviado',
      body: [
        'O app não exibe anúncios nem inclui SDK de publicidade.',
        'O Pro é uma compra única (não consumível). O pagamento é processado pelo Google Play, e seus dados de pagamento são tratados conforme a política de privacidade do Google.',
        'Para saber se você já tem o Pro, o app consulta a Play Store do seu aparelho. O próprio app não envia nada aos servidores do desenvolvedor nem a qualquer outro lugar, nem mesmo dados de compra ou identificadores. Ele nunca envia seu nome, e-mail ou dados da lista, nem envia informações para publicidade, análise ou qualquer outro fim.',
        `Política de privacidade do Google: ${GOOGLE_PRIVACY}`,
      ],
    },
    {
      title: 'Desenvolvedor e contato',
      body: [`Desenvolvedor: ${DEVELOPER}`, `Para dúvidas sobre esta política ou o tratamento de dados, abra uma issue no GitHub: ${ISSUES_URL}`],
    },
  ],
}

// ページ・画面の見出し。スマホ版の COPY（mobile/lib/copy）もこれを参照する
export const PRIVACY_TITLE: Record<AppLanguage, string> = {
  ja: 'プライバシーポリシー',
  en: 'Privacy policy',
  ko: '개인정보 처리방침',
  es: 'Política de privacidad',
  de: 'Datenschutzerklärung',
  'pt-BR': 'Política de privacidade',
}
