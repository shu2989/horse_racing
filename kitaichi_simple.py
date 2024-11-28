import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from math import pi

# 評価項目の数値化
rating_to_value = {
    'S': 1.0,
    'A': 0.7,
    'B': 0.4,
    'C': 0.1
}

# 各評価項目の重み（総和が1.0になるように設定）
weights = {
    '能力': 0.3,
    '馬場': 0.05,
    'コース': 0.1,
    '展開': 0.2,
    '調教': 0.05,
    '血統': 0.15,
    '枠番': 0.15
}

# 払い戻し率の設定
payout_rate = 0.70

st.set_page_config(
    page_title='期待値計算',
    layout='wide',
    initial_sidebar_state='auto'
)

# Streamlitアプリの設定
st.sidebar.title('期待値計算')
st.sidebar.markdown('#### 各馬の期待値を計算し、ランキング化します')
st.title("競馬予想期待値計算ツール")

# CSVファイルのアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    # CSVファイルの読み込み
    df = pd.read_csv(uploaded_file)
    # st.markdown("### アップロードされたデータ:")
    st.write(df)

    # 各評価項目を数値化
    for column in weights.keys():
        df[column] = df[column].map(rating_to_value)

    # レーダーチャート表示用のタブ作成
    st.markdown('### 各馬評価')
    name_tabs = st.tabs(df['馬名'].tolist())
        
    for i, tab in enumerate(name_tabs):
        with tab:
            horse_data = df.iloc[i][list(weights.keys())].values
                
            # レーダーチャート用の設定
            num_vars = len(weights)
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]
                
            values = horse_data.tolist()
            values += values[:1]
                
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='red', alpha=0.25)
            ax.plot(angles, values, color='red', linewidth=2)

            # 項目の順番を「能力」が北に来るように調整
            labels = list(weights.keys())
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=12)

            # グラフのタイトル
            ax.set_title(df['馬名'][i], size=15, color='red', y=1.1)
            
            # Y軸の目盛りを設定（0.1刻み、最大値1.0）
            ax.set_ylim(0, 1.0)
            # ax.set_yticks([0.1 * i for i in range(1, 11)])  # 0.1刻みの目盛り
            ax.set_yticks([0.1, 0.4, 0.7, 1.0])
            ax.set_yticklabels([])
                
            st.pyplot(fig)
    
    # 計算ボタン
    if st.button("計算"):
        st.divider()
        # 各馬の1着になる確率の計算
        df['好走率'] = 0
        for column, weight in weights.items():
            df['好走率'] += df[column] * weight
        
        # 0%～70%のレンジに正規化
        df['好走率'] = df['好走率'] * payout_rate

        # リスク係数の計算（オッズが高いほどリスクが高い）
        df['リスク係数'] = 1 / np.sqrt(df['単勝オッズ'])

        # リスク係数を考慮した勝率
        df['リスク調整後好走率'] = df['好走率'] * df['リスク係数']
        
        # 単勝支持率の計算
        # df['単勝支持率'] = payout_rate / df['単勝オッズ']
        
        # 期待値の計算
        # df['期待値'] = df['単勝オッズ'] * df['好走率']

        # 各馬の確率の総和を計算
        total_probability = df['リスク調整後好走率'].sum()

        # 100%に正規化
        df['正規化好走率'] = df['リスク調整後好走率'] / total_probability * 100
        
        # 期待値の再計算
        # df['期待値'] = df['単勝オッズ'] * df['正規化好走率']

        # 期待値の減衰計算
        df['調整後期待値'] = df['単勝オッズ'] * df['正規化好走率'] / np.log1p(df['単勝オッズ']) * 3

        # df['期待値'] = df['単勝オッズ'] * df['正規化好走率']

        # 期待値ランキングの表示
        df = df.sort_values(by='調整後期待値', ascending=False)
        
        st.markdown("### 期待値ランキング表")
        st.write(df[['馬名', '馬番', '単勝オッズ', '正規化好走率', '調整後期待値']])


        # 正規化勝率の棒グラフ
        fig, ax = plt.subplots(figsize=(12, 6))
        df_sorted_prob = df.sort_values(by='正規化好走率', ascending=False)
        sns.barplot(x='馬名', y='正規化好走率', data=df_sorted_prob, palette='Blues_d')
        ax.set_title("正規化好走率", fontsize=16)
        ax.set_xlabel("馬名", fontsize=12)
        ax.set_ylabel("正規化好走率 (%)", fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # グラフを表示
        st.markdown("### 好走率ランキング図")
        st.pyplot(fig)

        st.divider()

        # 期待値の棒グラフ
        df_sorted_value = df.sort_values(by='調整後期待値', ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='馬名', y='調整後期待値', data=df_sorted_value, palette='Reds_d')
        # 期待値100の部分に破線を追加
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1)
        ax.set_title("（調整後）期待値", fontsize=16)
        ax.set_xlabel("馬名", fontsize=12)
        ax.set_ylabel("調整後期待値", fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # グラフを表示
        st.markdown("### 期待値ランキング図")
        st.pyplot(fig)
