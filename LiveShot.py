import streamlit as st
from google import genai
import json
from PIL import Image
import pandas as pd
from datetime import datetime
import urllib.parse
import re
from streamlit_gsheets import GSheetsConnection
from google.cloud import storage
import uuid
import io
from google.oauth2 import service_account

def sanitize_filename(filename: str) -> str:
    # 半角スペースを _
    name = filename.replace(" ", "_")

    # URL・ファイルパス的に危険な文字を除去
    name = re.sub(r'[\\/:*?"<>|!()]+', "_", name)

    return name


def upload_image_to_storage(img_obj, filename):
    # Secretsから認証情報を再構築
    creds_info = st.secrets["gcp_service_account"]
    client = storage.Client.from_service_account_info(creds_info)

    bucket_name = "liveshot-image.firebasestorage.app"
    bucket = client.bucket(bucket_name)
    
    # ✅ ファイル名を安全化（/ やスペース問題を回避）
    safe_filename = sanitize_filename(filename)
    # ファイル名衝突防止
    unique_name = f"images/{uuid.uuid4()}_{safe_filename}"

    blob = bucket.blob(unique_name)
    blob.content_type = "image/jpeg"

    buf = io.BytesIO()
    img_obj.save(buf, format="JPEG")
    buf.seek(0)
    blob.upload_from_file(buf)

    image_url = f"https://storage.googleapis.com/{bucket_name}/{unique_name}"

    # デバッグ用（最初は一度出すと安心）
    st.write("✅ uploaded to:", unique_name)
    st.write("✅ image_url:", image_url)

    return image_url


#  - スキーム無し (www.example.com) も https:// を補完
def normalize_url(value):
    # None / NaN をはじく
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except  Exception:
        pass
    s = str(value).strip()

    # 空文字・"nan" 文字列を弾く
    if not s or s.lower() == "nan":
        return None
    
    # スキーム補完
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "https://" + s

    return s



# --- カレンダーURL生成ヘルパー ---
def get_google_calendar_url(name, date_str, venue, start_time, url):
    base_url = "https://www.google.com/calendar/render?action=TEMPLATE"
    
    # 日付と時刻の整形 (YYYYMMDD形式)
    try:
        clean_date = date_str.replace("/", "").replace("-", "")[:8]
    except:
        clean_date = ""

    # 時刻の整形 (19:00 -> 190000)
    time_str = "000000" # デフォルト
    if start_time:
        digits = re.sub(r'\D', '', start_time) # 数字のみ抽出
        if len(digits) >= 4:
            time_str = digits[:4] + "00"

    # 開始を設定
    start_dt = f"{clean_date}T{time_str}"
    safe_url = normalize_url(url) or ""

    params = {
        "text": f"ライブ: {name}",
        "dates": f"{start_dt}/{start_dt}",
        "location": venue,
        "details": f"詳細URL {safe_url}",
    }
    return base_url + "&" + urllib.parse.urlencode(params)

# --- 0. スプレッドシート接続設定 ---
# SQLiteの init_db の代わりにこれを使います
conn = st.connection("gsheets", type=GSheetsConnection)

# --- 1. 利用可能なモデルを取得する関数 ---
def get_available_models():
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        # モデル一覧を取得
        models = client.models.list()
        # 生成（generateContent）が可能なモデルのみを抽出
        valid_models = [
            (m.name or "").replace("models/", "")
            for m in models
            if "generateContent" in (getattr(m, "supported_actions", []) or [])
        ]
        # 取得できなかったときの保険（空配列対策）
        return valid_models or ["gemini-flash-lite-latest"]
    except Exception as e:
        st.error(f"モデルリストの取得に失敗しました: {e}")
        return ["gemini-flash-lite-latest"]

# --- 2. Gemini APIによる解析エンジン（model_idを引数で受け取る） ---
def extract_info_from_gemini(image, model_id):
    # 1. クライアントの初期化（configではなくClientオブジェクトを作る形式に）
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    # 今日の日付を取得してプロンプトに混ぜる
    current_year = datetime.now().year

    prompt = f"""
    あなたはライブ情報の抽出に特化したアシスタントです。
    提供された画像から、以下の項目を抽出し、JSON形式のみで出力してください。
    Markdownの装飾(```jsonなど)は一切不要です。
    
    【項目】
    - イベント名称
    - 公演日
      -公演日に関するルール
      -YYYY/MM/DD形式を推測
      -公演日に「年」の記載がない場合は、一律「{current_year}年」として補完
    - 会場名
    - 出演者リスト（配列）
    - 開演時間
    - チケット金額
    - 主催者
    - 問い合わせ先
    - 関連URL
    """

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, image]
        )

        # 4. JSONのパース（新しいSDKでは .text で結果が取れます）
        res_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(res_text)
    
    except Exception as e:
        # クォータエラーの場合、ユーザーにわかりやすく伝える
        if "429" in str(e):
            st.error(f"モデル '{model_id}' の無料枠制限に達しました。別のモデルを試すか、数分待ってください。")
        else:
            st.error(f"解析中にエラーが発生しました：{e}")
        return None

# --- 3. Streamlit UIレイアウト ---
st.set_page_config(page_title="LiveShot Admin", layout="wide")
st.title("🎸 LiveShot: スクショを予定に変える")

with st.sidebar:
    # サイドバーでモデルを選択
    st.header("⚙️ 設定")
    available_models = get_available_models()
    selected_model = st.selectbox(
        "使用するAIモデルを選択", 
        available_models,
        index=available_models.index("gemini-flash-lite-latest") if "gemini-flash-lite-latest" in available_models else 0
    )
    st.info(f"現在の選択: {selected_model}")

    st.divider()

    # サイドバー: 保存済みのデータの確認
    st.header("🗑️ データ管理")
    if st.button("全データを削除 (注意!)"):
        if st.checkbox("本当に削除しますか？"):
            # スプレッドシートを空のヘッダーのみにする処理
            empty_df  = pd.DataFrame(columns=['name', 'date', 'venue',
                                              'artists', 'start_time',
                                              'price', 'organizer',
                                              'contact', 'url'])
            conn.update(data=empty_df)
            st.warning("データを全削除しました")
            st.rerun()

# メインエリアをタブで分割
tab_register, tab_list = st.tabs(["🆕 ライブ登録", "📅 予定一覧・カレンダー"])

# --- タブ1: ライブ登録 ---
with tab_register:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. スクショをアップロード")
        uploaded_file = st.file_uploader("画像を選択．．．", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='解析対象の画像', width="stretch")

            if st.button("✨ AIで解析を実行する", type="primary"):
                with st.spinner(f'{selected_model} で解析中...'):
                    result = extract_info_from_gemini(image, selected_model)
                    if result:
                        st.session_state['edit_data'] = result
                        st.success("解析に成功しました！右側で内容を確認してください。")

    with col2:
        st.subheader("2. 内容の確認・保存")
        if 'edit_data' in st.session_state:
            d = st.session_state.get('edit_data', {})

            with st.form("edit_form"):
                name = st.text_input("イベント名称", d.get("イベント名称", ""))
                date = st.text_input("公演日", d.get("公演日", ""))
                venue = st.text_input("会場名", d.get("会場名", ""))
                # リストを編集可能な文字列に変換
                artists_list = d.get("出演者リスト", [])
                if not isinstance(artists_list, list):
                    artists_list = []
                artists = st.text_area("出演者（カンマ区切り）", ", ".join(artists_list) if artists_list else "")

                col_a, col_b = st.columns(2)
                with col_a:
                    start_time = st.text_input("開演時間", d.get("開演時間", ""))
                    price = st.text_input("チケット金額", d.get("チケット金額", ""))
                with col_b:
                    organizer = st.text_input("主催者", d.get("主催者", ""))
                    contact = st.text_input("問い合わせ先", d.get("問い合わせ先", ""))
                
                url = st.text_input("関連URL", d.get("関連URL", ""))

                if st.form_submit_button("✅ スプレッドシートに保存"):
                    # 画像をアップロードしてURLを取得
                    image_url = upload_image_to_storage(image, f"{name}_{date}.jpg")
                    safe_url = normalize_url(url) or ""
                    new_data = pd.DataFrame([{
                        "name": name,
                        "date": date,
                        "venue": venue,
                        "artists": artists,
                        "start_time": start_time,
                        "price": price,
                        "organizer": organizer,
                        "contact": contact,
                        "url": safe_url,
                        "image_url": image_url
                    }])

                    # 既存のデータを読み込んで結合
                    existing_data = conn.read(ttl=0) # ttl=0で最新を取得
                    updated_df = pd.concat([existing_data, new_data], ignore_index=True)
                    # 書き込み
                    conn.update(data=updated_df)

                    # セッションデータのクリーンアップ（安全な方法）
                    st.session_state.pop('edit_data', None)
                    st.success("保存しました！")
                    # 画面を再起動してリストを更新する
                    st.rerun()
                else:
                    st.info("左側で画像をアップロードして解析ボタンを押すと、ここに編集フォームが表示されます。")

# --- タブ2: 予定一覧・カレンダー ---
with tab_list:
    st.subheader("📅 保存されたライブ予定")

    # データ読み込み
    df_all = conn.read(ttl=0)

    if not df_all.empty:
        # 日付文字列をソート可能な形式に変換する補助（エラー回避のため）
        def safe_parse_date(date_str):
            try:
                # 2026/04/25 のような形式を想定
                dt = pd.to_datetime(date_str.replace("/", "-"))
                if dt.tzinfo is not None:
                    dt = dt.tz_localize(None)
                return dt
            except:
                return pd.NaT
            
        df_all['parsed_date'] = df_all['date'].apply(safe_parse_date)
        df_display = df_all.sort_values('parsed_date', ascending=True)

        # カレンダー風のタイムライン表示
        for idx, row in df_display.iterrows():
            # 編集モードかどうかの判定
            is_editing = st.session_state.get('editing_id') == idx
        
            with st.expander(f"📌 {row['date']} | {row['name']} @ {row['venue']}"):
                if is_editing:
                    # --- 編集フォームの表示 ---
                    with st.form(f"edit_{idx}"):
                        new_name = st.text_input("イベント名称", row['name'])
                        new_date = st.text_input("公演日", row['date'])
                        new_venue = st.text_input("会場名", row['venue'])
                        new_artists = st.text_area("出演者（カンマ区切り）", row['artists'])
                        new_start = st.text_input("開演時間", row['start_time'])
                        new_price = st.text_input("料金", row['price'])
                        new_organizer = st.text_input("主催者", row.get("organizer", ""))
                        new_contact = st.text_input("問い合わせ先", row.get("contact", ""))
                        new_url_raw = st.text_input("URL", row.get("url", ""))
                        new_url = normalize_url(new_url_raw) or ""

                        if "image_url" in row and pd.notna(row["image_url"]):
                            st.image(row["image_url"], caption="元のスクショ", width="stretch")

                        col_btn1, col_btn2 = st.columns(2)
                        if col_btn1.form_submit_button("💾 変更を保存"):
                            # parsed_date は表示用なので落としてから保存する
                            base_df = df_all.drop(columns=["parsed_date"])
                            base_df.loc[idx, ["name", "date", "venue", "artists", "start_time",
                                              "price", "organizer", "contact", "url"]] = [
                                                new_name,
                                                new_date,
                                                new_venue,
                                                new_artists,
                                                new_start,
                                                new_price,
                                                new_organizer,
                                                new_contact,
                                                new_url,
                                                ]
                            conn.update(data=base_df)
                            st.session_state.pop('editing_id', None) # 編集モード終了
                            st.success("更新しました！")
                            st.rerun()
                        
                        if col_btn2.form_submit_button("✖ キャンセル"):
                            st.session_state.pop('editing_id', None)
                            st.rerun()
                else:
                    # --- 通常表示 ---
                    col_info, col_artists = st.columns([2, 1])
                    with col_info:
                        st.write(f"**会場:** {row['venue']}")
                        st.write(f"**開演:** {row['start_time']}")
                        st.write(f"**料金:** {row['price']}")
                        cal_url = get_google_calendar_url(row['name'], row['date'], row['venue'], row['start_time'], row['url'])
                        st.link_button("📅 Googleカレンダーに登録", cal_url)
                        ticket_url = normalize_url(row.get("url"))
                        if ticket_url:
                            st.link_button("チケット・詳細URL", ticket_url)

                    with col_artists:
                        st.write("**出演:**")
                        artists_text = str(row.get("artists", "") or "")
                        for artist in artists_text.split(","):
                            a = artist.strip()
                            if a:
                                st.code(a)

                    st.divider()
                    col_edit, col_del = st.columns([1, 1])
                    if col_edit.button(f"📝 編集", key=f"edit_btn_{idx}"):
                        st.session_state['editing_id'] = idx
                        st.rerun()
                    
                    if col_del.button("🗑️ 削除", key=f"del_{idx}"):
                        base_df = df_all.drop(columns=["parsed_date"], errors="ignore")
                        base_df = base_df.drop(idx)
                        conn.update(data=base_df)
                        st.rerun()

    else:
        st.write("予定がありません。")