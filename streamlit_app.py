
import os
import re
import io
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="YouTube 키워드 분석 & 베스트셀러 랭킹",
    page_icon="📊",
    layout="wide",
)

DEFAULT_QUERIES = [
    "돈키호테 추천",
    "돈키호테 쇼핑리스트",
    "돈키호테",
    "일본 쇼핑 추천",
    "일본 드럭스토어 추천",
    "일본 편의점 추천",
    "일본 편의점",
    "일본 여행 쇼핑",
    "일본 기념품",
    "일본 기념품 추천",
    "일본 화장품 추천",
    "일본 화장품",
    "일본 간식",
    "일본 간식 추천",
]

DEFAULT_PRODUCTS = [
    "캔메이크", "CANMAKE", "Canmake", "키스미", "KISSME", "세잔느", "CEZANNE", "Cezanne",
    "하다라보", "HADALABO", "시세이도", "SHISEIDO", "비오레", "Biore", "마죠리카", "MAJOLICA",
    "파우더", "마스카라", "엣코스메", "틴트", "마스크팩", "헤어", "하이라이터",
    "섀도우", "베이스", "치크", "아이라이너", "크림", "립밤",
    "동전파스", "휴족시간", "아이봉", "산테FX", "로토Z", "오타이산", "페어아크네",
    "이브퀵", "파브론", "안약", "로토", "마츠키요",
    "곤약젤리", "자가리코", "킷캣", "도쿄바나나", "이치란", "후리카케",
    "일본 카레", "푸딩", "우마이봉", "로이스", "칼디", "JONETZ", "젤리", "말차", "컵라면",
    "치이카와", "산리오", "포켓몬", "짱구", "도라에몽", "리락쿠마", "스미코구라시", "건담", "원피스",
]

STOPWORDS = [
    "일본", "추천", "쇼핑", "돈키호테", "브이로그", "여행", "구매",
    "리뷰", "하울", "가격", "진짜", "좋은", "좋아요", "입니다",
    "그리고", "제품", "사용", "영상", "오늘", "이번", "소개",
    "제가", "저는", "너무", "정말", "그냥", "하면", "해서",
    "있는", "없는", "같아요", "합니다", "있습니다", "여러분",
    "후쿠오카", "도쿄", "오사카", "추천템", "드럭스토어", "일본여행", "쇼핑리스트",
]

ORDER_OPTIONS = {
    "관련도순": "relevance",
    "조회순": "viewCount",
    "최신순": "date",
}


def init_state():
    if "queries" not in st.session_state:
        st.session_state.queries = DEFAULT_QUERIES.copy()
    if "products" not in st.session_state:
        st.session_state.products = DEFAULT_PRODUCTS.copy()
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if "candidate_df" not in st.session_state:
        st.session_state.candidate_df = pd.DataFrame()
    if "mention_df" not in st.session_state:
        st.session_state.mention_df = pd.DataFrame()
    if "product_rank" not in st.session_state:
        st.session_state.product_rank = pd.DataFrame()


init_state()


# -----------------------------
# 유튜브 수집 함수
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_youtube_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def search_youtube_videos(youtube, query, order="relevance", max_results=50, region_code="KR"):
    video_ids = []
    request = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=max_results,
        regionCode=region_code,
        relevanceLanguage="ko",
        order=order,
    )
    response = request.execute()
    for item in response.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        if video_id:
            video_ids.append(video_id)
    return video_ids


def get_video_details(youtube, video_ids):
    results = []
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i + 50]
        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch_ids),
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            results.append({
                "video_id": item.get("id"),
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "tags": snippet.get("tags", []),
                "published_at": snippet.get("publishedAt", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
                "comment_count": int(stats.get("commentCount", 0)) if "commentCount" in stats else 0,
            })
    return results


def get_top_comments(youtube, video_id, max_comments=20):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            textFormat="plainText",
            order="relevance",
        )
        response = request.execute()
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(snippet.get("textDisplay", ""))
    except Exception:
        return ""
    return " ".join(comments)


def get_transcript_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["ko", "en", "ja"],
        )
        return " ".join([t["text"] for t in transcript])
    except Exception:
        return ""


# -----------------------------
# 분석 함수
# -----------------------------
def extract_hashtags(text):
    if pd.isna(text):
        return []
    return re.findall(r"#\w+", str(text))


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9#\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_candidate_df(df, stopwords=None):
    stopwords = stopwords or STOPWORDS
    all_text = " ".join(df["clean_text"].fillna(""))
    words = all_text.split()

    candidate_words = [
        w for w in words
        if (
            len(w) >= 2
            and w not in stopwords
            and not w.isdigit()
            and "http" not in w.lower()
            and "www" not in w.lower()
            and "com" not in w.lower()
        )
    ]

    word_rank = Counter(candidate_words).most_common()
    return pd.DataFrame(word_rank, columns=["keyword", "count"])


def enrich_text_columns(df):
    if df.empty:
        return df

    df = df.copy()
    df["hashtags_from_description"] = df["description"].apply(extract_hashtags)
    df["all_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["tags"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
        df.get("comments", "").fillna("") + " " +
        df.get("transcript", "").fillna("")
    )
    df["clean_text"] = df["all_text"].apply(clean_text)
    return df


def build_product_rank(df, products):
    mention_list = []

    if df.empty or not products:
        return pd.DataFrame(), pd.DataFrame()

    for _, row in df.iterrows():
        text = str(row.get("clean_text", "")).lower()
        for product in products:
            product = product.strip()
            if not product:
                continue

            count = text.count(product.lower())
            if count > 0:
                mention_list.append({
                    "product": product,
                    "count": count,
                    "video_id": row.get("video_id"),
                    "title": row.get("title"),
                    "view_count": row.get("view_count", 0),
                    "like_count": row.get("like_count", 0),
                    "comment_count": row.get("comment_count", 0),
                    "published_at": row.get("published_at"),
                    "channel_title": row.get("channel_title"),
                })

    mention_df = pd.DataFrame(mention_list)

    if mention_df.empty:
        return mention_df, pd.DataFrame()

    product_rank = (
        mention_df
        .groupby("product")
        .agg(
            mention_count=("count", "sum"),
            video_count=("video_id", "nunique"),
            total_view_count=("view_count", "sum"),
            total_like_count=("like_count", "sum"),
            total_comment_count=("comment_count", "sum"),
        )
        .reset_index()
    )

    max_view = product_rank["total_view_count"].max()
    product_rank["view_score"] = 0 if max_view == 0 else (product_rank["total_view_count"] / max_view) * 100
    product_rank["sns_score"] = (
        product_rank["mention_count"] * 0.5 +
        product_rank["video_count"] * 0.3 +
        product_rank["view_score"] * 0.2
    )

    product_rank = product_rank.sort_values("sns_score", ascending=False).reset_index(drop=True)
    product_rank.insert(0, "rank", product_rank.index + 1)

    return mention_df, product_rank


def make_wordcloud(candidate_df):
    if candidate_df.empty:
        return None

    word_freq = dict(zip(candidate_df["keyword"], candidate_df["count"]))
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    wc = WordCloud(
        font_path=font_path,
        width=1200,
        height=700,
        background_color="white",
        max_words=200,
        collocations=False,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def to_csv_download(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.title("📌 메뉴")
page = st.sidebar.radio(
    "페이지 선택",
    ["YouTube 키워드 분석", "베스트셀러 랭킹"],
)

st.sidebar.divider()
api_key = st.sidebar.text_input(
    "YouTube Data API Key",
    value=os.getenv("YOUTUBE_API_KEY", ""),
    type="password",
    help="로컬 실행 시 .env 또는 환경변수 YOUTUBE_API_KEY로 넣어도 됩니다.",
)

st.sidebar.caption("API 키는 코드에 직접 저장하지 않는 것을 권장합니다.")


# -----------------------------
# 공통 UI: 검색어/제품 사전 관리
# -----------------------------
def query_manager():
    st.subheader("1. 검색 키워드 관리")

    with st.form("add_query_form", clear_on_submit=True):
        new_query = st.text_input("새 검색어 추가", placeholder="예: 일본 돈키호테 필수템")
        submitted = st.form_submit_button("검색어 추가")
        if submitted and new_query.strip():
            if new_query.strip() not in st.session_state.queries:
                st.session_state.queries.append(new_query.strip())
                st.success(f"'{new_query.strip()}' 추가 완료")
            else:
                st.warning("이미 있는 검색어입니다.")

    selected_to_delete = st.multiselect(
        "삭제할 검색어 선택",
        options=st.session_state.queries,
    )
    if st.button("선택한 검색어 삭제", disabled=not selected_to_delete):
        st.session_state.queries = [q for q in st.session_state.queries if q not in selected_to_delete]
        st.success("선택한 검색어를 삭제했습니다.")
        st.rerun()

    st.write("현재 검색어")
    st.dataframe(pd.DataFrame({"queries": st.session_state.queries}), use_container_width=True, hide_index=True)


def product_manager():
    st.subheader("3. 제품명 사전 관리")

    with st.form("add_product_form", clear_on_submit=True):
        new_product = st.text_input("새 제품명 추가", placeholder="예: 무인양품, 메구리즘, DHC")
        submitted = st.form_submit_button("제품명 추가")
        if submitted and new_product.strip():
            if new_product.strip() not in st.session_state.products:
                st.session_state.products.append(new_product.strip())
                st.success(f"'{new_product.strip()}' 추가 완료")
            else:
                st.warning("이미 있는 제품명입니다.")

    selected_to_delete = st.multiselect(
        "삭제할 제품명 선택",
        options=st.session_state.products,
    )
    if st.button("선택한 제품명 삭제", disabled=not selected_to_delete):
        st.session_state.products = [p for p in st.session_state.products if p not in selected_to_delete]
        st.success("선택한 제품명을 삭제했습니다.")
        st.rerun()

    st.write("현재 제품명 사전")
    st.dataframe(pd.DataFrame({"products": st.session_state.products}), use_container_width=True, hide_index=True)


# -----------------------------
# 페이지 1: YouTube 키워드 분석
# -----------------------------
if page == "YouTube 키워드 분석":
    st.title("📊 YouTube 키워드 분석 대시보드")
    st.caption("검색어를 추가/삭제하고, 정렬 옵션을 바꿔 유튜브 영상·댓글·자막 기반 키워드 후보를 수집합니다.")

    left, right = st.columns([1, 2], gap="large")

    with left:
        query_manager()

    with right:
        st.subheader("2. 크롤링 옵션")

        col1, col2, col3 = st.columns(3)
        with col1:
            order_label = st.selectbox("정렬 기준", list(ORDER_OPTIONS.keys()), index=0)
            order = ORDER_OPTIONS[order_label]
        with col2:
            max_results = st.slider("검색어별 영상 수", 5, 50, 30, step=5)
        with col3:
            region_code = st.selectbox("지역 코드", ["KR", "JP", "US"], index=0)

        collect_comments = st.checkbox("댓글 수집", value=True)
        collect_transcripts = st.checkbox("자막 수집", value=True)
        max_comments = st.slider("영상별 댓글 수", 5, 100, 20, step=5, disabled=not collect_comments)

        if st.button("🚀 크롤링 업데이트", type="primary"):
            if not api_key:
                st.error("사이드바에 YouTube Data API Key를 입력해주세요.")
            elif not st.session_state.queries:
                st.error("검색어를 1개 이상 추가해주세요.")
            else:
                try:
                    youtube = get_youtube_client(api_key)

                    progress = st.progress(0)
                    status = st.empty()
                    all_video_ids = []

                    for idx, query in enumerate(st.session_state.queries, start=1):
                        status.write(f"검색 중: {query}")
                        ids = search_youtube_videos(
                            youtube,
                            query=query,
                            order=order,
                            max_results=max_results,
                            region_code=region_code,
                        )
                        all_video_ids.extend(ids)
                        progress.progress(idx / max(len(st.session_state.queries), 1))

                    all_video_ids = list(dict.fromkeys(all_video_ids))
                    status.write(f"고유 영상 {len(all_video_ids)}개 상세정보 수집 중...")

                    video_data = get_video_details(youtube, all_video_ids)
                    df = pd.DataFrame(video_data)

                    if not df.empty:
                        if collect_comments:
                            comments = []
                            for i, vid in enumerate(df["video_id"], start=1):
                                status.write(f"댓글 수집 중: {i}/{len(df)}")
                                comments.append(get_top_comments(youtube, vid, max_comments=max_comments))
                            df["comments"] = comments
                        else:
                            df["comments"] = ""

                        if collect_transcripts:
                            transcripts = []
                            for i, vid in enumerate(df["video_id"], start=1):
                                status.write(f"자막 수집 중: {i}/{len(df)}")
                                transcripts.append(get_transcript_text(vid))
                            df["transcript"] = transcripts
                        else:
                            df["transcript"] = ""

                        df = enrich_text_columns(df)
                        candidate_df = build_candidate_df(df)

                        st.session_state.df = df
                        st.session_state.candidate_df = candidate_df
                        st.session_state.mention_df = pd.DataFrame()
                        st.session_state.product_rank = pd.DataFrame()

                    status.success("크롤링 및 키워드 후보 업데이트 완료")
                    progress.progress(1.0)

                except HttpError as e:
                    st.error(f"YouTube API 오류: {e}")
                except Exception as e:
                    st.error(f"실행 중 오류가 발생했습니다: {e}")

    st.divider()

    df = st.session_state.df
    candidate_df = st.session_state.candidate_df

    if df.empty:
        st.info("아직 수집된 데이터가 없습니다. 검색어와 옵션을 설정한 뒤 '크롤링 업데이트'를 눌러주세요.")
    else:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("수집 영상 수", f"{len(df):,}")
        kpi2.metric("총 조회수", f"{int(df['view_count'].sum()):,}")
        kpi3.metric("자막 있는 영상", f"{int((df['transcript'] != '').sum()):,}")
        kpi4.metric("댓글 수집 영상", f"{int((df['comments'] != '').sum()):,}")

        st.subheader("수집 영상 데이터")
        st.dataframe(
            df[["title", "channel_title", "published_at", "view_count", "like_count", "comment_count", "video_id"]],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "영상 데이터 CSV 다운로드",
            data=to_csv_download(df),
            file_name="youtube_video_data.csv",
            mime="text/csv",
        )

        st.subheader("키워드 후보 TOP 100")
        top_n = st.slider("표시할 키워드 수", 10, 200, 100, step=10)
        st.dataframe(candidate_df.head(top_n), use_container_width=True, hide_index=True)
        st.download_button(
            "candidate_df CSV 다운로드",
            data=to_csv_download(candidate_df),
            file_name="candidate_df.csv",
            mime="text/csv",
        )

        st.subheader("워드 클라우드")
        fig = make_wordcloud(candidate_df.head(top_n))
        if fig:
            st.pyplot(fig, clear_figure=True)


# -----------------------------
# 페이지 2: 베스트셀러 랭킹
# -----------------------------
else:
    st.title("🏆 제품별 베스트셀러 후보 랭킹")
    st.caption("YouTube 수집 데이터(candidate_df 업데이트 이후)를 기반으로 제품명 사전에 포함된 단어의 언급량·영상수·조회수를 합산해 랭킹화합니다.")

    left, right = st.columns([1, 2], gap="large")

    with left:
        product_manager()

    with right:
        st.subheader("랭킹 생성")
        st.write("크롤링 업데이트로 생성된 영상 데이터에서 제품명 사전 단어를 찾아 제품별 점수를 계산합니다.")

        if st.button("🏆 랭킹 보기", type="primary"):
            if st.session_state.df.empty:
                st.error("먼저 'YouTube 키워드 분석' 페이지에서 크롤링 업데이트를 실행해주세요.")
            else:
                mention_df, product_rank = build_product_rank(st.session_state.df, st.session_state.products)
                st.session_state.mention_df = mention_df
                st.session_state.product_rank = product_rank

                if product_rank.empty:
                    st.warning("제품명 사전에 해당하는 언급이 발견되지 않았습니다.")
                else:
                    st.success("제품별 랭킹 생성 완료")

    st.divider()

    product_rank = st.session_state.product_rank
    mention_df = st.session_state.mention_df

    if product_rank.empty:
        st.info("아직 생성된 랭킹이 없습니다. 제품명 사전을 수정한 뒤 '랭킹 보기'를 눌러주세요.")
    else:
        st.subheader("제품별 랭킹")
        st.dataframe(product_rank, use_container_width=True, hide_index=True)
        st.download_button(
            "제품 랭킹 CSV 다운로드",
            data=to_csv_download(product_rank),
            file_name="product_rank.csv",
            mime="text/csv",
        )

        st.subheader("TOP 20 차트")
        chart_df = product_rank.head(20).set_index("product")[["sns_score", "mention_count", "video_count"]]
        st.bar_chart(chart_df)

        with st.expander("제품 언급 상세 데이터 보기"):
            st.dataframe(mention_df, use_container_width=True, hide_index=True)
            st.download_button(
                "제품 언급 상세 CSV 다운로드",
                data=to_csv_download(mention_df),
                file_name="mention_df.csv",
                mime="text/csv",
            )
