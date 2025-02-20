GOOGLE_API_KEY = "AIzaSyCpd2U1ReQD2NwyX9vC-1XjLmLQOOOSM1o"

import streamlit as st
import os
import google.generativeai as genai

st.title("ChatGPT-like clone")
genai.configure(api_key=GOOGLE_API_KEY)

st.title("Hello")

# サイドバーにGeminiのAPIキーの入力欄を設ける
with st.sidebar:
    gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
    "[Get an Gemini API key](https://aistudio.google.com/app/apikey)"
    "[View the source code](https://github.com/danishi/streamlit-gemini-chatbot)"

st.title("✨ Gemini Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Gemini")

# PDF読み込みボタンの追加
uploaded_pdf = st.file_uploader("PDFを読み込む", type=["pdf"])
if uploaded_pdf is not None:
    import PyPDF2
    pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() or ""
    st.write("### PDFの内容:")
    st.text_area("PDF Content", pdf_text, height=300)

# セッションにチャット履歴がなければ初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# チャット履歴を表示
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# ユーザーの入力が送信された際に実行される処理
if prompt := st.chat_input("How can I help you?"):

    # APIキーのチェック
    if not gemini_api_key:
        st.info("Please add your [Gemini API key](https://aistudio.google.com/app/apikey) to continue.")
        st.stop()

    # モデルのセットアップ
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    # ユーザの入力をチャット履歴に追加し画面表示
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # これまでの会話履歴を取得
    messages = []
    for message in st.session_state.chat_history:
        messages.append({
            "role": message["role"] if message["role"] == "user" else "model",
            'parts': message["content"]
        })

    # Geminiへ問い合わせを行う
    response = model.generate_content(messages)

    # Geminiの返答をチャット履歴に追加し画面表示
    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
    st.chat_message("assistant").write(response.text)
