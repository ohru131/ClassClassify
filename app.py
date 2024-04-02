import itertools
import os
from base64 import b64encode
from itertools import combinations

import pandas as pd
import streamlit as st
from amplify import VariableGenerator, FixstarsClient, solve


def create_student_pairs_dataframe(column_name1, column_name2, pairs, student_dict):
    """
    生徒番号のペアを使って、出席番号と名前のペアを含むDataFrameを作成する。

    Args:
        column_name1 (str): 最初のペアに使用する列名
        column_name2 (str): 2番目のペアに使用する列名
        pairs (list): 生徒番号のペアのリスト
        student_dict (dict): 生徒番号と名前の辞書

    Returns:
        DataFrame: 出席番号と名前のペアを含むDataFrame
    """
    student_pairs1 = []
    student_pairs2 = []
    for pair in pairs:
        student1 = pair[0]
        student2 = pair[1]
        name1 = student_dict.get(student1, "???")
        name2 = student_dict.get(student2, "???")
        student_pairs1.append(f"{student1}:{name1}")
        student_pairs2.append(f"{student2}:{name2}")
    return pd.DataFrame({column_name1: student_pairs1, column_name2: student_pairs2})


def check_solution(solutions, wanted_pairs, unwanted_pairs, student_dict):
    """
    ソリューションをチェックし、問題のペアが正しく解決されたかどうかを確認する。

    Args:
    - solutions (DataFrame): ソリューションを表すDataFrame
    - wanted_pairs (list): 欲しいペアのリスト
    - unwanted_pairs (list): 望ましくないペアのリスト
    - student_dict (dict): 生徒番号と名前の辞書

    Returns:
    - list: チェック結果を示すメッセージのリスト
    """
    result = []

    for pair in wanted_pairs:
        if not any(solutions[pair[0], k] == 1 and solutions[pair[1], k] == 1 for k in range(solutions.shape[1])):
            name1 = student_dict.get(pair[0], "???")
            name2 = student_dict.get(pair[1], "???")
            result.append(f"同じ組失敗　{pair[0]}:{name1} と{pair[1]}:{name2} が別の組になっています。")

    for pair in unwanted_pairs:
        if any(solutions[pair[0], k] == 1 and solutions[pair[1], k] == 1 for k in range(solutions.shape[1])):
            name1 = student_dict.get(pair[0], "???")
            name2 = student_dict.get(pair[1], "???")
            result.append(f"別の組ペアの生徒 {pair[0]}:{name1} と生徒 {pair[1]}:{name2} が同じ組になっています。")

    if not result:
        result.append("なし")

    return result


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_pairs(xlsx_file_path, sheet_name):
    df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, header=None)
    pairs = []
    for _, row in df.iterrows():
        # NaNを除外して、行のすべての値を数値に変換
        students = [int(float(x)) for x in row if pd.notnull(x) and is_number(str(x).strip())]
        # すべてのペアの組み合わせを作成
        pairs.extend(combinations(students, 2))
    return pairs


def read_num_classes(xlsx_file_path, sheet_name="設定"):
    try:
        # Excelファイルを読み込む
        df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, header=None)

        # B3セルの値を取得して整数値に変換
        num_classes = int(df.iloc[2, 1])

        return num_classes
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None


def read_student_data(xlsx_file_path, sheet_name="生徒名簿"):
    # CSVファイルを読み込む（header=Noneで列名行を無視）
    # original_df = pd.read_csv(csv_file_path, header=None)
    original_df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, header=None)

    one_hot_df = pd.DataFrame()

    # 重み係数を読み込み、数値に変換（セルの値を文字列に変換してから処理を行う 変換できない場合は1を代入）
    weight_coefficients = [float(x) if str(x).replace('.', '', 1).isdigit() else 1 for x in
                           original_df.iloc[0, 2:].values]
    # 項目名を読み込む
    column_names = original_df.iloc[1, 2:].values

    # データ部分を抽出し、列名を変更
    data_df = original_df.iloc[2:].copy()
    data_df.columns = list(original_df.iloc[1, :])

    weight_dict = {}
    for col, weight in zip(column_names, weight_coefficients):
        # 空白やNullを0に置き換え
        data_df[col].fillna("", inplace=True)

        # unique_valuesを取得（すべての値を文字列に変換してからソート）
        try:
            unique_values = sorted([str(x) for x in data_df[col].unique()])
        except Exception as e:
            print(e)

        # 列の値が2項目だけの場合は列名そのまま、値が2項目目と等しい場合は1にする
        if len(unique_values) == 2:
            new_col_name = f"{col}_{unique_values[1]}"
            one_hot_df[new_col_name] = data_df[col].apply(lambda x: 1 if str(x) == unique_values[1] else 0)
            weight_dict[new_col_name] = weight
        else:
            # 列の値が3種類以上ある場合、列名_値名
            for idx, value in enumerate(unique_values, start=1):
                new_col_name = f"{col}_{value}"
                one_hot_df[new_col_name] = data_df[col].apply(lambda x: 1 if str(x) == value else 0)
                weight_dict[new_col_name] = weight

    data_dic = {col: one_hot_df[col].values for col in one_hot_df.columns}
    header_list = one_hot_df.columns.tolist()
    weight_list = [weight_dict[col] for col in header_list]

    # 生徒番号と名前の辞書を作成
    student_numbers = data_df["NO"].tolist()
    student_dict = dict(zip(student_numbers, data_df["名前"]))

    return original_df, one_hot_df, [data_dic[col] for col in
                                     one_hot_df.columns], header_list, weight_list, student_dict


def compute_penalty(num_students, num_classes, unwanted_pairs, wanted_pairs, lam, x):
    # 同じ生徒は複数のクラスに所属しない
    penalty = lam * sum((sum(x[i, k] for k in range(num_classes)) - 1) ** 2 for i in range(num_students))
    # クラスの人数は同じくらい
    penalty += lam * sum(
        (sum(x[i, k] for i in range(num_students)) - num_students / num_classes) ** 2 for k in range(num_classes))
    # 別のクラスにしたいペア
    # 各ペアについて、その両方の生徒が同じクラスに所属する場合にのみペナルティを加える
    for pair in unwanted_pairs:
        for k in range(num_classes):
            penalty += lam * x[pair[0], k] * x[pair[1], k]
    # 同じクラスにしたいペア
    # 各ペアについて、その両方の生徒が同じクラスに所属しない場合にペナルティを加える
    for pair in wanted_pairs:
        for k in range(num_classes):
            penalty -= lam * (1 - x[pair[0], k]) * (1 - x[pair[1], k])
    return penalty


def compute_cost(num_students, num_classes, weight_list, data_list, x):
    cost = 0
    for i, (w, weight) in enumerate(zip(data_list, weight_list)):
        cost += weight * 1 / num_classes * sum(
            (sum(w[j] * x[j, k] for j in range(num_students)) - 1 / num_classes * sum(
                sum(w[j] * x[j, k] for j in range(num_students)) for k in range(num_classes))) ** 2 for k in
            range(num_classes))
    return cost


# ロジックを含む関数
def solve_problem(token, num_classes, weight_list, data_list, unwanted_pairs, wanted_pairs):
    if data_list:
        num_students = len(data_list[0])
    else:
        num_students = 0

    gen = VariableGenerator()  # 変数のジェネレータを宣言
    x = gen.array("Binary", shape=(num_students, num_classes))  # 決定変数を作成
    # x = gen.array(num_students, num_classes)

    lam1 = 10
    y = compute_cost(num_students, num_classes, weight_list, data_list, x)
    y += compute_penalty(num_students, num_classes, unwanted_pairs, wanted_pairs, lam1, x)

    # 実行マシンクライアントの設定
    client = FixstarsClient()
    client.token = token
    client.parameters.timeout = 1000
    result = solve(y, client)  # 問題を入力してマシンを実行

    # 解の存在の確認
    if len(result) == 0:
        raise RuntimeError("The given constraints are not satisfied")

    # 結果の取得
    values = result[0].values  # 解を格納
    solutions = x.evaluate(values)
    return solutions


def create_solution_df(xlsx_file_path, solutions, one_hot_df):
    original_df = pd.read_excel(xlsx_file_path, sheet_name="生徒名簿", header=None)

    solution_df = original_df.iloc[2:, :2].copy()
    solution_df.reset_index(drop=True, inplace=True)

    for k in range(solutions.shape[1]):
        solution_df[f"クラス{k+1}"] = ["○" if solutions[i, k] == 1 else "" for i in range(solutions.shape[0])]

    header_row = pd.DataFrame([["No.", "名前"] + [f"クラス{k+1}" for k in range(solutions.shape[1])]],
                              columns=solution_df.columns)

    solution_df = pd.concat([header_row, solution_df])

    class_assignments = pd.DataFrame(solutions, columns=[k for k in range(solutions.shape[1])])
    one_hot_df_with_class = one_hot_df.copy().reset_index(drop=True)
    one_hot_df_with_class['クラス'] = class_assignments.idxmax(axis=1) + 1

    aggregated_df = one_hot_df_with_class.groupby('クラス').sum().reset_index()

    class_counts = pd.Series(class_assignments.sum(), name="人数")
    aggregated_df.insert(1, "人数", class_counts)

    # Create a DataFrame for each class
    class_dfs = []
    for k in range(solutions.shape[1]):
        class_df = original_df.iloc[2:, :].copy()
        class_df.columns = original_df.iloc[1]
        class_df = class_df[solutions[:, k] == 1]
        class_dfs.append(class_df)

    return solution_df, aggregated_df, class_dfs


class ClassOptimizer:
    def __init__(self, xlsx_file_path, token):
        # データ読み込み
        num_classes = read_num_classes(xlsx_file_path, sheet_name="設定")
        (self.original_df,
         self.one_hot_df,
         data_list,
         header_list,
         weight_list,
         student_dict) = read_student_data(xlsx_file_path, sheet_name="生徒名簿")
        wanted_pairs = read_pairs(xlsx_file_path, sheet_name="同じ組ペア")
        unwanted_pairs = read_pairs(xlsx_file_path, sheet_name="別の組ペア")

        # DataFrameを作成する
        self.wanted_pairs_df = create_student_pairs_dataframe("生徒1","生徒2", wanted_pairs, student_dict)
        self.unwanted_pairs_df = create_student_pairs_dataframe("生徒1","生徒2", unwanted_pairs, student_dict)

        # 最適化
        solutions = solve_problem(token, num_classes, weight_list, data_list, unwanted_pairs,
                                  wanted_pairs)
        # 　結果検証
        (self.solution_df,
         self.aggregated_df,
         self.class_dfs) = create_solution_df(xlsx_file_path, solutions, self.one_hot_df)
        # Check solution and create DataFrame for failed combinations
        failed_combinations = check_solution(solutions, wanted_pairs, unwanted_pairs, student_dict)
        self.failed_combinations_df = pd.DataFrame(failed_combinations, columns=["組み合わせ失敗"])


# Streamlitの処理
def app():
    # token = "AE/DA9enVyvhM3Y2SANsMCTLZKg9gTKmv23" # ご自身のトークンを入力
    encrypted_token = b')$_(@\x1eY7"\x0c)7\n#?\'\x08\x1c* \x00R\x04=19\x04\x1e\x08.*R;8!'

    def xor_cypher(input_string, key):
        return ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(input_string, itertools.cycle(key)))

    def download_zip_file(zip_file_path, zip_file_name):
        with open(zip_file_path, "rb") as f:
            zip_file_bytes = f.read()
        st.download_button(
            label="Download ZIP File",
            data=zip_file_bytes,
            file_name=zip_file_name,
            mime="application/zip"
        )

    def download_solution(class_optimize: ClassOptimizer, filename='solution.xlsx'):
        with pd.ExcelWriter(filename) as writer:
            class_optimize.original_df.to_excel(writer, sheet_name="生徒名簿", header=False, index=False)
            class_optimize.wanted_pairs_df.to_excel(writer, sheet_name="同じ組ペア", header=False, index=False)
            class_optimize.unwanted_pairs_df.to_excel(writer, sheet_name="別の組ペア", header=False, index=False)
            class_optimize.solution_df.to_excel(writer, sheet_name="クラス分け", header=False, index=False)
            for i, class_df in enumerate(class_optimize.class_dfs):
                class_df.to_excel(writer, sheet_name=f"クラス{i+1}", header=True, index=False)
            class_optimize.aggregated_df.to_excel(writer, sheet_name="集計", index=False)
            class_optimize.failed_combinations_df.to_excel(writer, sheet_name="組み合わせ失敗", header=False,
                                                           index=False)

        with open(filename, 'rb') as f:
            data = f.read()


        b64 = b64encode(data).decode()
        st.markdown(f'''
        <a href="data:file/xlsx;base64,{b64}" download="{filename}">
            クラス分け結果のダウンロード
        </a>
        ''', unsafe_allow_html=True)

    st.title("クラス分けソフト")
    st.write("生徒の特性が各クラスでできるだけ均等になるように、クラス分けを行います")

    zip_file_name = "template.zip"
    zip_file_path = os.path.join(os.path.dirname(__file__), zip_file_name)

    st.write("生徒名簿のひな形Excelファイルをダウンロードし、加筆修正してください")
    download_zip_file(zip_file_path, zip_file_name)

    password = st.text_input("パスワードを入力してください")  # ユーザーがパスワードを入力
    if password == "":
        exit()

    decrypted_token = xor_cypher(encrypted_token.decode(),password)  # トークンを復号化

    # 復号化したトークンを使用
    token = decrypted_token
    uploaded_file = st.file_uploader("xlsxファイルをアップロードしてください", type=["xlsx"])

    if uploaded_file is not None:
        with st.spinner("ファイルを処理中..."):
            # if num_classes != 0:
            try:
                class_optimize = ClassOptimizer(uploaded_file, token)
            except RuntimeError as e:
                if '401: Unauthorized' in str(e):
                    st.write("パスワードが違うため、処理できませんでした。")
                    exit()
                else:
                    raise e

            st.write("アップロードされたExcelファイルの内容:")
            st.write(class_optimize.original_df)
            st.write("同じ組ペア")
            st.write(class_optimize.wanted_pairs_df)
            st.write("別の組ペア")
            st.write(class_optimize.unwanted_pairs_df)
            st.write(class_optimize.one_hot_df)

            st.write("結果表示:")
            st.write(class_optimize.solution_df)
            st.write(class_optimize.aggregated_df)
            st.write(class_optimize.failed_combinations_df)
            for i, class_df in enumerate(class_optimize.class_dfs):
                st.write(f"クラス{i+1}")
                st.write(class_df)

            st.write("結果のダウンロード:")
            download_solution(class_optimize,
                              filename='クラス分け結果.xlsx')


if __name__ == "__main__":
    app()
