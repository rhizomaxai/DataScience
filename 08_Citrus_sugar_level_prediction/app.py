"""제주 감귤 당도 예측 — 프로젝트 ② 결과 웹앱 (8차시)
실행: streamlit run app.py
데이터: data/seongsan_weather.csv (기상청, cp949), data/citrus_brix.csv (교재 제공, utf-8)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

st.set_page_config(page_title="제주 감귤 당도 예측", page_icon="🍊", layout="wide")

DATA = Path(__file__).parent / "data"          # 앱 파일 옆의 data/ 폴더 (상대 경로)
FEATURES = ["최고기온", "최저기온", "가조시간"]
COLORS = {"main": "#E76F51", "sub": "#2A9D8F", "gray": "#5B8E9F", "accent": "#F4A259", "dark": "#264653"}
MONTH_WEATHER = {  # 측정일 날씨의 월평균 (탐색 결과)
    "8월": (31.5, 25.6, 13.6), "9월": (28.2, 23.2, 12.6), "10월": (24.5, 17.6, 11.6), "11월": (19.1, 10.2, 10.6)}


# ------------------------------------------------------------------ 데이터 준비
@st.cache_data
def load_raw():
    weather = pd.read_csv(DATA / "seongsan_weather.csv", encoding="cp949")
    brix = pd.read_csv(DATA / "citrus_brix.csv")
    return weather, brix


@st.cache_data
def preprocess():
    weather_raw, brix_raw = load_raw()
    w = weather_raw.drop(["지점", "지점명"], axis=1)
    w["일시"] = pd.to_datetime(w["일시"])
    w = w.set_index("일시")
    w.columns = ["평균기온", "최저기온", "최고기온", "일강수량", "가조시간"]
    missing = w.isna().sum()
    w["평균기온"] = w["평균기온"].fillna((w["최저기온"] + w["최고기온"]) / 2)
    w = w.fillna(0)

    b = brix_raw.copy()
    b["일시"] = pd.to_datetime(b["일시"])
    b = b.set_index("일시")
    b_all = b.copy()
    b = b[b.index.month != 12]
    merged = w.merge(b, left_index=True, right_index=True, how="inner")
    merged["일차"] = merged.index.dayofyear
    counts = {"당도 측정 전체": len(b_all), "12월 제외 후": len(b), "기상 자료와 합친 뒤": len(merged)}
    return w, b_all, merged, missing, counts


def poly_model(degree):
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())


@st.cache_data
def holdout_results(degree):
    _, _, merged, _, _ = preprocess()
    X, y = merged[FEATURES], merged["당도"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    m = poly_model(degree).fit(X_tr, y_tr)
    pred = m.predict(X_te)
    return {"학습 R²": r2_score(y_tr, m.predict(X_tr)), "평가 R²": r2_score(y_te, pred),
            "MSE": mean_squared_error(y_te, pred), "MAE": mean_absolute_error(y_te, pred),
            "기준선 MAE": mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean())),
            "y_test": y_te.values, "pred": pred, "dates": y_te.index, "n_train": len(X_tr), "n_test": len(X_te)}


@st.cache_data
def cv_results(cols=tuple(FEATURES), degrees=(1, 2, 3, 4, 5)):
    _, _, merged, _, _ = preprocess()
    X, y = merged[list(cols)], merged["당도"]
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    rows = []
    for d in degrees:
        r2 = cross_val_score(poly_model(d), X, y, cv=kf, scoring="r2")
        mse = -cross_val_score(poly_model(d), X, y, cv=kf, scoring="neg_mean_squared_error")
        mae = -cross_val_score(poly_model(d), X, y, cv=kf, scoring="neg_mean_absolute_error")
        rows.append({"차수": d, "평균 R²": r2.mean(), "평균 MSE": mse.mean(), "평균 MAE": mae.mean(), "묶음별 R²": r2})
    return pd.DataFrame(rows)


@st.cache_resource
def final_model():
    _, _, merged, _, _ = preprocess()
    return poly_model(2).fit(merged[FEATURES], merged["당도"])


weather, brix_all, merged, missing, counts = preprocess()
cv_main = cv_results()
final = final_model()
best_cv = cv_main.loc[cv_main["차수"] == 2].iloc[0]

# ------------------------------------------------------------------ 사이드바
st.sidebar.title("🍊 제주 감귤 당도 예측")
st.sidebar.caption("데이터분석 프로젝트 ② 최종 결과")
PAGES = ["🏠 한눈에 보기", "① 문제 정의", "② 데이터 출처", "③ 탐색 및 전처리",
         "④ 머신러닝 모델과 선택 이유", "⑤ 모델 평가", "⑥ 개선사항 및 활용"]
page = st.sidebar.radio("페이지", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(f"""**최종 모델**
표준화 → 2차 다항 특성 → 선형 회귀
특성: 최고기온 · 최저기온 · 가조시간
교차 검증 R² **{best_cv['평균 R²']:.3f}** · MAE **{best_cv['평균 MAE']:.2f}°Bx**""")
st.sidebar.caption("데이터: 기상청 기상자료개방포털(성산) · 교재 2부 프로젝트 2 감귤 당도 측정 데이터")


def section_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)


# ------------------------------------------------------------------ 페이지: 한눈에 보기
if page == PAGES[0]:
    section_header("제주 감귤 당도 예측하기", "날씨 데이터(기온·가조시간)로 감귤의 당도(°Bx)를 예측한 프로젝트의 최종 결과")
    c = st.columns(4)
    monthly = merged.groupby(merged.index.month)["당도"].mean()
    kpis = [("분석 데이터", f"{len(merged)}개", "당도 측정일 × 그날의 날씨"),
            ("교차 검증 R²", f"{best_cv['평균 R²']:.3f}", "2차 다항 회귀 · 5겹 교차 검증"),
            ("평균 오차 (MAE)", f"{best_cv['평균 MAE']:.2f} °Bx", "교차 검증 평균"),
            ("8월 → 11월 당도", f"{monthly.loc[8]:.1f} → {monthly.loc[11]:.1f}", "익을수록 달아진다 (°Bx)")]
    for col, (label, value, cap) in zip(c, kpis):
        with col.container(border=True):
            st.metric(label, value)
            st.caption(cap)

    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(x=[f"{m}월" for m in monthly.index], y=monthly.values.round(2), text=monthly.values.round(1),
                     labels={"x": "", "y": "평균 당도(°Bx)"}, title="월별 평균 당도 (2015~2025 측정일)",
                     color_discrete_sequence=[COLORS["accent"]])
        fig.update_layout(height=360, yaxis_range=[0, 11])
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("핵심 결과 세 줄")
        st.markdown(f"""
1. **가조시간(낮의 길이)** 이 당도와 가장 관계가 깊다 (상관계수 **{merged.corr().loc['가조시간', '당도']:.2f}**).
2. **교차 검증**으로 고른 2차 다항 회귀가 평균 **{best_cv['평균 MAE']:.2f}°Bx** 오차로 당도를 예측한다 (평균으로 찍기보다 훨씬 작다).
3. 하지만 **날짜 하나**만으로도 비슷한 성능이 나온다 → 모델이 배운 것은 '날씨'보다 **'계절'** 에 가깝다.
""")
        st.info("왼쪽 메뉴에서 단계별 내용을, **⑥ 개선사항 및 활용**에서 직접 당도를 예측해 볼 수 있습니다.")

# ------------------------------------------------------------------ 페이지: 문제 정의
elif page == PAGES[1]:
    section_header("① 문제 정의", "배경 · 목적 · 분석 질문 · 머신러닝 문제")
    a, b = st.columns(2)
    with a:
        st.subheader("배경")
        st.markdown("""
- 감귤은 나무에 달린 동안 **익으면서 달아진다**. 농가는 당도를 보고 수확·출하 시기를 정한다.
- 당도를 재려면 열매를 따서 즙을 짜 **직접 측정**해야 한다.
- 반면 **날씨 데이터**는 기상청 관측소에서 매일 자동으로 쌓인다.
""")
        st.subheader("목적")
        st.markdown("""
- 매일 쌓이는 날씨 데이터로 **당도를 미리 가늠**할 수 있는지 확인한다.
- 당도와 관계가 깊은 기상 요소를 찾고, 그 관계를 **정직하게 해석**한다.
""")
    with b:
        st.subheader("분석 질문")
        st.success("그날의 날씨(기온·햇빛)로 감귤의 당도를 예측할 수 있을까?")
        st.subheader("머신러닝 문제로 바꾸기")
        st.table(pd.DataFrame({
            "구분": ["입력 X (특성)", "정답 y (타깃)", "문제 종류", "누가 쓸까?"],
            "내용": ["최고기온 · 최저기온 · 가조시간 (3개)", "당도 (°Bx, 과즙 100g 속 당의 g 수)",
                   "지도학습 · 회귀 (특성이 여러 개 → 다중 회귀)", "감귤 농가 · 농업기술센터 (수확 시기 가늠)"]}).set_index("구분"))
        st.caption("가조시간 = 해가 떠서 질 때까지의 시간(낮의 길이). 구름이 껴도 줄지 않고 날짜·위치로 정해진다.")

# ------------------------------------------------------------------ 페이지: 데이터 출처
elif page == PAGES[2]:
    section_header("② 데이터 출처", "두 개의 데이터를 날짜로 합쳐 사용했다")
    weather_raw, brix_raw = load_raw()
    st.table(pd.DataFrame({
        "": ["출처", "기간", "간격 · 크기", "주요 열", "파일 · 형식"],
        "기상 자료": ["기상청 기상자료개방포털 · 종관기상관측(ASOS) 성산 관측소(188)",
                  f"{weather_raw['일시'].min()} ~ {weather_raw['일시'].max()}", f"매일 · {len(weather_raw):,}일",
                  "평균·최저·최고기온(℃), 일강수량(mm), 가조시간(hr)", "data/seongsan_weather.csv · cp949"],
        "당도 자료": ["교재 『실습으로 배우는 데이터 과학』 2부 프로젝트 2 제공",
                  f"{brix_raw['일시'].min()} ~ {brix_raw['일시'].max()}", f"약 보름 간격 · {len(brix_raw)}회 (8~12월)",
                  "일시, 당도(°Bx)", "data/citrus_brix.csv · utf-8"]}).set_index(""))
    t1, t2 = st.tabs(["기상 자료 원본 미리 보기", "당도 자료 원본 미리 보기"])
    t1.dataframe(weather_raw.head(10), width="stretch")
    t2.dataframe(brix_raw.head(10), width="stretch")
    st.warning("두 데이터는 **간격(매일 vs 보름)** 과 **기간**이 다르다. 공통 열은 '일시'(날짜) 하나뿐이라 같은 날짜끼리 합쳤다.")

# ------------------------------------------------------------------ 페이지: 탐색 및 전처리
elif page == PAGES[3]:
    section_header("③ 탐색 및 전처리", "두 데이터를 손질하고, 합치고, 관계를 살펴봤다")
    tab1, tab2, tab3, tab4 = st.tabs(["3-1 기상 자료 손질", "3-2 당도 자료 탐색", "3-3 두 데이터 합치기", "3-4 상관 탐색"])

    with tab1:
        st.markdown("**전처리 순서**: 지점·지점명 삭제 → 일시를 날짜로 바꿔 인덱스로 → 열 이름 정리 → 결측치 처리")
        a, b = st.columns([2, 3])
        with a:
            st.markdown("**결측치는 이유에 맞게 채웠다**")
            st.table(pd.DataFrame({
                "결측 수": [int(missing["평균기온"]), int(missing["일강수량"])],
                "이유": ["그날 기록이 빠짐 (2025-05-28)", "비가 안 온 날은 값을 적지 않음"],
                "처리": ["(최저기온 + 최고기온) ÷ 2", "0으로 채움"]}, index=["평균기온", "일강수량"]))
            st.caption("모두 0으로 채웠다면 5월 28일 평균기온이 0℃가 되어 데이터가 망가진다.")
        with b:
            yearly = weather.resample("YE").agg({"평균기온": "mean", "일강수량": "sum"})
            yearly.index = yearly.index.year
            choice = st.radio("연도별로 보기", ["연 강수량(mm)", "연평균 기온(℃)"], horizontal=True)
            col = "일강수량" if choice.startswith("연 강수량") else "평균기온"
            fig = px.bar(yearly, x=yearly.index.astype(str), y=col, labels={"x": "연도", col: choice},
                         color_discrete_sequence=[COLORS["gray"]])
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")
            st.caption("2015년 값이 튀는 이유: 기상 자료가 2015-11-15부터라 한 달 반치뿐이다.")
        fig = px.box(weather[["평균기온", "최저기온", "최고기온", "가조시간"]].melt(var_name="요소", value_name="값"),
                     x="요소", y="값", color="요소", title="기온·가조시간 분포 (가조시간은 9.9~14.4시간으로 좁다)")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        pivot = brix_all.groupby([brix_all.index.year, brix_all.index.month])["당도"].mean().rename_axis(["연도", "월"]).reset_index()
        years = sorted(pivot["연도"].unique())
        pick = st.multiselect("연도 고르기", years, default=years)
        fig = px.line(pivot[pivot["연도"].isin(pick)], x="월", y="당도", color=pivot[pivot["연도"].isin(pick)]["연도"].astype(str),
                      markers=True, labels={"color": "연도", "당도": "당도(°Bx)"}, title="월별 당도 변화 (선 하나 = 한 해)")
        fig.update_layout(height=380, xaxis=dict(tickmode="array", tickvals=[8, 9, 10, 11, 12]))
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        a.info("거의 모든 해에 **8월 → 11월로 당도가 오른다** (익어 간다). 2018년은 8월부터 9.1로 높게 시작했다.")
        b.warning("**12월 측정은 2022년 단 1회(11.18)** 뿐이라 다른 해와 비교할 수 없어 분석에서 제외했다.")

    with tab3:
        a, b = st.columns([3, 2])
        with a:
            fig = go.Figure(go.Funnel(y=list(counts.keys()), x=list(counts.values()), textinfo="value",
                                      marker={"color": [COLORS["gray"], COLORS["accent"], COLORS["main"]]}))
            fig.update_layout(height=300, title="행 수의 변화")
            st.plotly_chart(fig, width="stretch")
        with b:
            st.code("merged = weather.merge(brix, left_index=True,\n                       right_index=True, how='inner')", language="python")
            st.markdown(f"""
- `how='inner'`: 양쪽 모두에 있는 날짜만 남긴다 → 당도를 잰 날의 날씨가 붙는다
- **{counts['12월 제외 후']} → {counts['기상 자료와 합친 뒤']}행**: 기상 자료가 2015-11-15부터라, 2015년 8~11월 초 측정 {counts['12월 제외 후'] - counts['기상 자료와 합친 뒤']}회는 짝지을 날씨가 없어 빠졌다
""")
        st.dataframe(merged.drop(columns="일차").head(8), width="stretch")

    with tab4:
        corr = merged.drop(columns="일차").corr().round(2)
        a, b = st.columns(2)
        with a:
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="상관계수 히트맵")
            fig.update_layout(height=430)
            st.plotly_chart(fig, width="stretch")
        with b:
            feat = st.selectbox("당도와 비교할 요소", ["가조시간", "최저기온", "평균기온", "최고기온", "일강수량"])
            fig = px.scatter(merged, x=feat, y="당도", color=merged.index.month.astype(str),
                             labels={"color": "월", "당도": "당도(°Bx)"}, title=f"{feat}와 당도 (상관계수 {corr.loc[feat, '당도']:.2f})")
            fig.update_layout(height=380)
            st.plotly_chart(fig, width="stretch")
        st.warning("기온·가조시간과 당도는 **강한 음의 상관**(약 −0.8)이다. 하지만 8월→11월에는 기온↓·낮↓·익어 감↑이 "
                   "**함께** 일어난다. '추워서 달아진다'는 인과로 해석하면 안 된다 (→ ⑥에서 확인).")

# ------------------------------------------------------------------ 페이지: 모델과 이유
elif page == PAGES[4]:
    section_header("④ 사용할 머신러닝 모델과 선택 이유", "표준화 → 다항 특성(2차) → 선형 회귀")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="sans-serif", color="#264653", fillcolor="#EEF4F4"];
  X [label="최고기온\\n최저기온\\n가조시간"]; S [label="① 표준화\\nStandardScaler"]; P [label="② 2차 다항 특성\\nx, x², x·y … (9개)"];
  L [label="③ 선형 회귀\\nLinearRegression"]; Y [label="당도 예측\\n(°Bx)", fillcolor="#F4A259"]; X->S->P->L->Y; }""")
    a, b = st.columns(2)
    with a:
        st.subheader("왜 이 모델인가?")
        st.markdown(f"""
1. **정답이 숫자** → 회귀 모델이 필요하다.
2. **데이터가 {len(merged)}개뿐** → 복잡한 모델은 금방 외워 버린다(과대적합). 단순한 선형 계열이 알맞다.
3. **해석이 쉽다** → 특성마다 기울기로 관계의 방향을 설명할 수 있다.
4. **약간의 곡선 관계** → 2차 다항 특성으로 직선보다 조금 유연하게 맞춘다.
5. **교차 검증 결과** → 1차({cv_main.loc[0, '평균 R²']:.3f})와 2차({cv_main.loc[1, '평균 R²']:.3f})가 가장 좋고 3차부터 떨어졌다.
""")
        st.subheader("왜 표준화를 먼저?")
        st.markdown("기온 33을 네제곱하면 약 119만처럼 큰 수가 되어 계산이 불안정해진다. 평균 0·표준편차 1로 맞추면 안정적이다.")
    with b:
        st.subheader("왜 이 세 특성인가?")
        corr = merged.corr()["당도"].drop(["당도", "일차"]).round(3).sort_values()
        fig = px.bar(x=corr.values, y=corr.index, orientation="h", text=corr.values, labels={"x": "당도와의 상관계수", "y": ""},
                     color=corr.index.isin(FEATURES), color_discrete_map={True: COLORS["main"], False: "#C9D3D5"})
        fig.update_layout(height=300, showlegend=False, xaxis_range=[-1, 0.3])
        st.plotly_chart(fig, width="stretch")
        st.caption("상관이 강한 기상 요소를 골랐다(교재 기준). 일강수량은 상관이 0.01로 거의 없어 제외했다. 평균기온은 최고·최저기온과 거의 같은 정보다.")
        X_tr, _, y_tr, _ = train_test_split(merged[FEATURES], merged["당도"], test_size=0.2, random_state=0)
        lin = LinearRegression().fit(X_tr, y_tr)
        st.markdown("**참고: 1차(다중 선형 회귀) 기울기**")
        st.table(pd.DataFrame({"기울기": lin.coef_.round(3)}, index=FEATURES))
        st.caption("특성끼리 상관이 0.9 안팎으로 닮아 있어, 가조시간이 몫을 대부분 가져간다 → 기울기를 '중요도'로 바로 읽지 않는다.")

# ------------------------------------------------------------------ 페이지: 평가
elif page == PAGES[5]:
    section_header("⑤ 모델 평가 방법 및 평가", "한 번 나누어 평가 + 5겹 교차 검증")
    st.markdown("""
| 평가 방법 | 내용 |
|---|---|
| **한 번 나누기** | 83개 중 80%(66개)로 학습, 20%(17개)로 시험 (`random_state=0`) |
| **5겹 교차 검증** | 5묶음으로 나눠 돌아가며 시험 · 점수 평균 (`KFold`, `random_state=2`) — 운의 영향을 줄인다 |
| **지표** | R²(1에 가까울수록 좋음) · MSE(오차² 평균) · MAE(평균 몇 °Bx 틀렸나) · 기준선(평균으로 찍기) |
""")
    st.subheader("5-1. 한 번 나누어 평가하기")
    deg = st.slider("다항 차수", 1, 4, 2)
    r = holdout_results(deg)
    c = st.columns(5)
    c[0].metric("학습 R²", f"{r['학습 R²']:.3f}")
    c[1].metric("평가 R²", f"{r['평가 R²']:.3f}")
    c[2].metric("MSE", f"{r['MSE']:.3f}")
    c[3].metric("MAE", f"{r['MAE']:.3f} °Bx")
    c[4].metric("기준선 MAE (평균으로 찍기)", f"{r['기준선 MAE']:.3f} °Bx")
    a, b = st.columns([3, 2])
    with a:
        fig = px.scatter(x=r["y_test"], y=r["pred"], labels={"x": "실제 당도", "y": "예측 당도"},
                         title=f"{deg}차 모델 · 실제 vs 예측 (평가용 {r['n_test']}개)", color_discrete_sequence=[COLORS["sub"]])
        fig.add_trace(go.Scatter(x=[5.5, 11], y=[5.5, 11], mode="lines", line=dict(color=COLORS["main"]), name="정확히 맞힘"))
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with b:
        rows = [holdout_results(d) for d in [1, 2, 3, 4]]
        st.dataframe(pd.DataFrame({"차수": [1, 2, 3, 4], "학습 R²": [x["학습 R²"] for x in rows], "평가 R²": [x["평가 R²"] for x in rows],
                                   "MSE": [x["MSE"] for x in rows]}).round(3).set_index("차수"), width="stretch")
        st.caption("한 번 나누면 3차가 가장 좋아 보이고, 4차는 학습 R²만 오르는 과대적합이다.")

    st.subheader("5-2. 5겹 교차 검증으로 다시 평가하기")
    a, b = st.columns(2)
    with a:
        show = cv_main.copy()
        show["평균 R²(표시)"] = show["평균 R²"].clip(lower=-0.2)
        fig = px.bar(show, x=show["차수"].astype(str), y="평균 R²(표시)", text=show["평균 R²"].round(3),
                     labels={"x": "차수", "평균 R²(표시)": "교차 검증 평균 R²"}, title="차수별 교차 검증 평균 R²",
                     color=show["차수"] == 2, color_discrete_map={True: COLORS["main"], False: COLORS["gray"]})
        fig.update_layout(height=360, showlegend=False, yaxis_range=[-0.25, 0.85])
        st.plotly_chart(fig, width="stretch")
        st.caption("5차는 R²가 크게 음수(붕괴)라 그래프에서는 −0.2에서 잘라 표시했다.")
    with b:
        folds = cv_main.loc[0, "묶음별 R²"]
        fig = px.bar(x=[f"시험 {i}" for i in range(1, 6)], y=folds.round(3), text=folds.round(2), labels={"x": "", "y": "R²"},
                     title=f"1차 모델의 묶음별 R² (평균 {folds.mean():.3f})", color_discrete_sequence=[COLORS["sub"]])
        fig.update_layout(height=360, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")
        st.caption("같은 모델인데 어떤 묶음으로 시험 보느냐에 따라 0.47~0.91로 흔들린다 → 점수 하나는 '운'이 섞인 숫자.")
    st.dataframe(cv_main.drop(columns="묶음별 R²").round(3).set_index("차수"), width="stretch")
    st.success(f"**결론**: 교차 검증에서는 1·2차가 가장 좋다(R² {cv_main.loc[0, '평균 R²']:.3f} · {cv_main.loc[1, '평균 R²']:.3f}). "
               f"한 번 나누기에서 최고였던 3차는 오히려 낮다. 최종 모델은 **2차**(평균 MAE {best_cv['평균 MAE']:.2f}°Bx)로 정했다.")

# ------------------------------------------------------------------ 페이지: 개선사항 및 활용
else:
    section_header("⑥ 개선사항 및 활용방법", "무엇을 개선했고, 무엇을 알게 되었으며, 어떻게 쓸 수 있는가")
    tab1, tab2, tab3 = st.tabs(["개선사항", "🍊 당도 예측해 보기", "활용방법과 한계"])

    with tab1:
        st.markdown("""
| 개선 | 내용 | 효과 |
|---|---|---|
| 결측치를 이유에 맞게 | 평균기온 → 최저·최고의 평균, 강수량 → 0 | 사실과 다른 값(5월에 0℃) 방지 |
| 12월 측정 제외 | 2022년 단 1회뿐 | 비교할 수 없는 값 제거 |
| 표준화 후 다항 특성 | `make_pipeline`으로 묶음 | 큰 수로 인한 계산 불안정 방지 |
| 교차 검증으로 차수 선택 | 한 번 나누기 → 5겹 교차 검증 | 3차(운 좋은 결과) 대신 2차 선택 |
""")
        st.subheader("🔍 반전: 날씨가 아니라 '날짜'였다?")
        comp = {"교재 3특성 (최고·최저기온·가조시간)": FEATURES, "가조시간 하나": ["가조시간"], "일차(날짜) 하나": ["일차"]}
        vals = {k: cv_results(tuple(v), (2,)).loc[0, "평균 R²"] for k, v in comp.items()}
        a, b = st.columns([3, 2])
        with a:
            fig = px.bar(x=list(vals.keys()), y=list(vals.values()), text=[f"{v:.3f}" for v in vals.values()],
                         labels={"x": "", "y": "2차 · 교차 검증 평균 R²"}, color=list(vals.keys()),
                         color_discrete_sequence=[COLORS["gray"], COLORS["accent"], COLORS["main"]])
            fig.update_layout(height=340, showlegend=False, yaxis_range=[0, 0.85])
            st.plotly_chart(fig, width="stretch")
        with b:
            st.markdown("""
'1년 중 며칠째'인지(**일차**) 하나만으로도 날씨 세 가지와 같거나 더 좋은 점수가 나왔다.

→ 모델이 배운 것은 '추우면 달다'가 아니라 사실상 **'가을이 깊어지면 달다'**.
기온·가조시간은 날짜를 대신 알려 준 **숨은 변수(계절)** 관계다.
""")
        st.subheader("앞으로 해 볼 개선")
        c = st.columns(3)
        c[0].markdown("**같은 날짜끼리 비교**  \n해마다 10월 초 당도만 모아 그해 날씨와 비교 → 계절 효과 제거")
        c[1].markdown("**익는 동안의 날씨**  \n측정 전 30일 평균기온·누적 강수량(`rolling`)을 특성으로")
        c[2].markdown("**날짜 + 날씨 함께**  \n일차에 날씨를 더해 점수가 오르면 그만큼이 '날씨의 몫'")

    with tab2:
        st.markdown("월을 고르면 그달의 대표 날씨(측정일 평균)가 채워진다. 슬라이더로 바꿔 보자.")
        month = st.selectbox("시기", list(MONTH_WEATHER.keys()), index=2)
        hi0, lo0, sun0 = MONTH_WEATHER[month]
        a, b, c = st.columns(3)
        hi = a.slider("최고기온 (℃)", 10.0, 36.0, hi0, 0.1, key=f"hi_{month}")
        lo = b.slider("최저기온 (℃)", 0.0, 30.0, lo0, 0.1, key=f"lo_{month}")
        sun = c.slider("가조시간 (시간)", 9.9, 14.4, sun0, 0.1, key=f"sun_{month}")
        if lo > hi:
            st.error("최저기온이 최고기온보다 높습니다. 값을 다시 확인해 주세요.")
        else:
            pred = float(final.predict(pd.DataFrame({"최고기온": [hi], "최저기온": [lo], "가조시간": [sun]}))[0])
            mae = best_cv["평균 MAE"]
            k = st.columns([1, 2])
            k[0].metric("예측 당도", f"{pred:.2f} °Bx")
            k[0].caption(f"오차 범위(±MAE {mae:.2f}): 약 {pred - mae:.1f} ~ {pred + mae:.1f} °Bx")
            with k[1]:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=round(pred, 2), number={"suffix": " °Bx"},
                                             gauge={"axis": {"range": [5, 11]}, "bar": {"color": COLORS["main"]},
                                                    "steps": [{"range": [5, 7], "color": "#EEF4F4"}, {"range": [7, 9], "color": "#FBE3D2"},
                                                              {"range": [9, 11], "color": "#F4A259"}]}))
                fig.update_layout(height=230, margin=dict(t=20, b=10))
                st.plotly_chart(fig, width="stretch")
            outside = not (merged["최고기온"].min() <= hi <= merged["최고기온"].max()
                           and merged["최저기온"].min() <= lo <= merged["최저기온"].max())
            if outside:
                st.warning("입력한 기온이 학습 데이터의 범위(8~11월 측정일) 밖이다 → 외삽이라 예측을 믿기 어렵다.")
            st.caption("예측은 '이 시기·날씨라면 대략 이 정도'라는 기대값이다. 개별 과수원의 실제 당도와는 다를 수 있다.")

    with tab3:
        a, b = st.columns(2)
        with a:
            st.subheader("✅ 이렇게 쓸 수 있다")
            st.markdown(f"""
- 이 시기·날씨라면 당도가 **대략 어느 정도**일지 가늠 (평균 ±{best_cv['평균 MAE']:.1f}°Bx)
- 지역 전체의 **당도 흐름**을 미리 살펴보고 측정 계획을 세우는 데 참고
- 날씨 데이터만 있으면 되므로 **매일 자동으로** 업데이트할 수 있다
""")
            st.subheader("⛔ 이렇게 쓰면 안 된다")
            st.markdown("""
- 개별 과수원의 수확 결정을 **이 숫자 하나로** 내리기
- "올해 날씨 **때문에** 달았다"는 **원인 설명** (모델은 사실상 계절을 배웠다)
""")
        with b:
            st.subheader("한계")
            st.markdown(f"""
- 데이터가 **{len(merged)}개**뿐이다
- **측정 당일 하루**의 날씨만 썼다 (익는 동안의 날씨가 아님)
- **관측소 한 곳**(성산)의 날씨다
- 당도를 잰 **과수원·품종 정보가 없다**
- **숨은 변수: 계절** — 상관을 인과로 해석하면 안 된다
""")
            st.info("결과를 전할 때는 **근거 · 제안 · 한계**를 함께 말한다.")

st.sidebar.divider()
st.sidebar.caption("데이터분석 기초 · 8차시 프로젝트 ② · Streamlit")
