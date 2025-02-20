
from tqdm import tqdm
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import time
import streamlit as st
import google.generativeai as genai
from pathlib import Path
from pdf2image import convert_from_path
from pathlib import Path
import os
import fitz 
from pathlib import Path
import time
from io import BytesIO
import os
from PIL import Image
import os
import google.generativeai as genai
st.title("DD with gemini-2.0-flash")
genai.configure(api_key=GOOGLE_API_KEY)


# サイドバーにGeminiのAPIキーの入力欄を設ける
with st.sidebar:
    gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
    "[Get an Gemini API key](https://aistudio.google.com/app/apikey)"
    "[View the source code](https://github.com/danishi/streamlit-gemini-chatbot)"

st.title("✨ Gemini Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Gemini")

pdf_folder = Path("/workspaces/streamit_sample_app/data/input")
pdf_folder.mkdir(parents=True, exist_ok=True)

# PDFファイルをアップロード
uploaded_pdf = st.file_uploader("PDFを読み込む", type=["pdf"])

if uploaded_pdf is not None:
    # 保存パスを設定
    save_path = pdf_folder / uploaded_pdf.name
    
    # PDFを保存
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())  # PDFファイルがあるフォルダ
image_folder = Path("/workspaces/streamit_sample_app/data/output")  # 画像ファイルを保存するフォルダ
image_folder.mkdir(exist_ok=True)  # 画像フォルダがなければ作成

# Open the first PDF file from the PDF folder
pdf_files = list(pdf_folder.glob("*.pdf"))  # Use the first PDF for example
if pdf_files:
    for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)

        # PDFごとのサブフォルダを作成
        pdf_subfolder = image_folder / "image" / pdf_file.stem
        pdf_subfolder.mkdir(parents=True, exist_ok=True)

        # 各ページを画像に変換
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            image_path = pdf_subfolder / f"{pdf_file.stem}_{page_num + 1}.jpg"
            pix.save(image_path)

        doc.close()
        st.success(f"画像が {pdf_subfolder} に保存されました。")
else:
    st.warning("PDFファイルが見つかりませんでした。")


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

import os
import google.generativeai as genai



# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

response.text.strip() if response and response.text else None
def process_with_gemini(image):
    """
    Send the page image to Gemini and extract structured Markdown content.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")

    # API Rate Limiting
    time.sleep(2)
    
    prompt = """
    Extract all contents from the provided PDF page image and format it as Markdown in Japanese.  
    Do NOT extract repeated headers that appear on every page.  

    ### Guidelines: 
    - Structure text properly into headings, paragraphs, lists, and bullet points.  
    - Tables: If a table exists, extract it accurately and format it using proper Markdown table syntax.  
    - Graphs & Figures: If a graph or figure contains important numerical data, summarize key insights and represent them as a table if possible.  
    - Equations: If mathematical formulas appear, format them using LaTeX syntax within Markdown.  
    """

    response = model.generate_content([prompt, image])
    return response.text.strip() if response and response.text else None
from pathlib import Path
from PIL import Image

# 画像フォルダと出力フォルダ

image_folder = Path("/workspaces/streamit_sample_app/data/output/image/定款_SBIホールディングス")
output_dir = Path("/workspaces/streamit_sample_app/data/output/text docs")
output_dir.mkdir(exist_ok=True)
combined_dir = Path("/workspaces/streamit_sample_app/data/output")
# 結合用のリスト
combined_content = []

# 画像ファイルを番号順に処理
image_paths = sorted(image_folder.glob("*.jpg"), key=lambda x: int(x.stem.split('_')[-1]))  # Sort by number in filename

for image_path in image_paths:
    try:
        # 画像を開いて処理
        img = Image.open(image_path).convert("RGB")  # RGBに変換
        
        # Geminiで処理
        markdown_content = process_with_gemini(img)
        
        # 内容があればリストに追加
        if markdown_content:
            combined_content.append(markdown_content)
            print(f"Successfully processed {image_path}")
        else:
            print(f"No content extracted from {image_path}")
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

# 最後にすべての内容を1つのファイルに保存
if combined_content:
    output_path = combined_dir / "combined_docs.md"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_content))  # コンテンツを結合して保存
        print(f"All markdown content has been successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving combined markdown: {str(e)}")
else:
    print("No content was extracted from any images.")
