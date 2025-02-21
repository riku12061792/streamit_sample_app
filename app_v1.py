
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
import json
# st.title("DD with gemini-2.0-flash")
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
        pdf_file = pdf_file
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
    pdf_file = pdf_file
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

# # 画像フォルダと出力フォルダ
image_folder = Path("/workspaces/streamit_sample_app/data/output/image")


pdf_subfolders = [f for f in image_folder.iterdir() if f.is_dir()]

# Use the stem of the PDF file as the folder name (or another variable)


combined_dir = Path("/workspaces/streamit_sample_app/data/output")
# 結合用のリスト
combined_content = []
if pdf_subfolders:
    pdf_subfolder = pdf_subfolders[0]
# 画像ファイルを番号順に処理
image_paths = sorted(pdf_subfolder.glob("*.jpg"), key=lambda x: int(x.stem.split('_')[-1]))  # Sort by number in filename

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



def cosine_similarity(vec_a, vec_b):
    # np.arrayに変換しておくと、ドット積やノルムの計算が簡単
    a = np.array(vec_a)
    b = np.array(vec_b)

    # コサイン類似度 = (a · b) / (||a|| * ||b||)
    # np.dot() で内積を計算し、np.linalg.norm() でベクトルの長さを計算
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter

input_folder = Path("/workspaces/streamit_sample_app/data/output")
output_folder = Path("/workspaces/streamit_sample_app/data/output/chunked_docs")
output_folder.mkdir(parents=True, exist_ok=True)  # Make sure chunked_docs folder exists

# チャンク設定
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=300
)

# インプットが1ファイルの場合、入力ファイルを取得
text_file = input_folder / "combined_docs.md"

# ファイルを処理
with open(text_file, "r", encoding="utf-8") as f:
    text = f.read()
    

# チャンク化
chunks = text_splitter.split_text(text)
# `chunked_docs` フォルダを作成
file_output_folder = output_folder  # Use the chunked_docs folder directly
file_output_folder.mkdir(parents=True, exist_ok=True)

# チャンクファイルを保存
for i, chunk in enumerate(chunks):
    output_file = file_output_folder / f"{text_file.stem}_chunk{i+1}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(chunk)
        print(len(chunk))

print(f"✅ {text_file.name} を {len(chunks)} チャンクに分割しました。フォルダ：{file_output_folder}")


# Define the folder path
output_folder = Path("/workspaces/streamit_sample_app/data/output/chunked_docs")
output_base_folder = Path("/workspaces/streamit_sample_app/data/output/vector_DB")  # Base folder to store embeddings
output_base_folder.mkdir(parents=True, exist_ok=True)

# List to store all embeddings
all_embeddings = []

# Loop through the files in the output folder
for file_path in output_folder.glob("*.txt"):
    # Read the content of the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()  # Read the content of the file
    
    print(f"Processing {file_path.name}... (Length of content: {len(content)})")
    
    # Generate the embedding for the content
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=content
    )
    
    # Assuming the response contains the embeddings in a field 'embedding'
    embeddings = response.get('embedding')
    
    if embeddings:
        # Collect embedding data for this file
        embedding_data = {
            "file_name": file_path.name,
            "content": content,
            "embedding": embeddings
        }
        
        # Append the embedding data to the list
        all_embeddings.append(embedding_data)
        
        print(f"✅ Embeddings for {file_path.name} have been added to the list.")

    # Optional: to avoid hitting rate limits or API restrictions, add a delay between requests
    # sleep(1)  # Adjust the delay as needed (in seconds)

# Save all embeddings to a single JSON file
final_embedding_json_path = output_base_folder / "final_embeddings.json"

# Save to a JSON file inside the base folder
with open(final_embedding_json_path, "w", encoding="utf-8") as json_file:
    json.dump(all_embeddings, json_file, ensure_ascii=False, indent=4)

print(f"✅ All embeddings have been saved to {final_embedding_json_path}")

def generate_answer(user_query, context):
    # API Rate Limiting
    # time.sleep(5)
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # # system message
    # system_content = (
    #     "<コンテキスト>のみを参考にして、質問に関連する情報をすべて取得してまとめてください。\n\n"
    #     "<注意点>\n"
    #     "- コンテキストのみから情報を抽出してください。\n"
    #     "- コンテキストに答えの手がかりが見つからないと判断される場合は、「分かりません」と答えること。\n"
    # )

    # user message
    user_content = (
        "<コンテキスト>のみを参考にして、質問に対する直接の回答のみ出力してください。"
        "回答作成は、文字数上限54を超えないでください。文字数を超える場合は、回答を短くしてください。"
        "コンテキストからの情報で回答作成が困難な場合は、「わかりません」と回答してください。以下の、注意点を考慮してください。\n\n"
        "<注意点>\n"
        "- コンテキストのみから回答を導け。\n"
        "- 小数第n位を四捨五入する時は、小数第n-1位までを回答としなさい。\n"
        "- 数値の回答に「,」は含めないでください。\n"
        "- 回答は直接の答えのみ出力してください。「。」や「です」なども回答に必要ありません。\n"
        "- 数量で答える問題の回答には、質問に記載の単位を使うこと（例：○○円、○○倍、○○%など）。\n"
        "- 数量で答える問題の回答には、数量と単位のみで回答してください。（例：○○円、○○倍、○○%など）。\n"
        "- コンテキストに答えの手がかりが見つからないと判断される場合はその旨を「分かりません」と答えること。\n\n"
        "<質問>\n"
        f"{user_query}\n\n"
        "<コンテキスト>\n"
        f"{context}\n\n"
    )

   # Generate response using Gemini model with custom parameters
    # response = model.generate_content(
    #     [user_content],
    #     max_tokens=30000,     # Maximum number of tokens in the response
    #     temperature=0,   # Adjust the temperature for randomness
    #     top_p=0.95,        # Nucleus sampling
    #     frequency_penalty=0,
    #     presence_penalty=0,
    # )
    response = model.generate_content([user_content])

    # print("\n===== 回答 =====\n")
    # print(response)
    return response.text.strip() if response and response.text else None
def prompt_engineering(user_query):
   
    user_content = (
        "<質問>\n"
        f"{user_query}\n\n"
        "<質問>を書き換えてください。例えば、質問から社名を省いたり、わかりやすくしたりしてください。なお、投資DDのための情報を取得するために質問文が使われます。"
        "すでに、そのドキュメントにはアクセスできると仮定して、情報検索のために質問文を簡潔かつ向上してください。書き換えられた質問文のみ出力してください。"
    )

    messages = [
        # {"role": "system", "content": system_content}, # system message
        {"role": "user", "content": user_content}, # user message
    ]

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([prompt])
    return response.text.strip() if response and response.text else None

def generate_context(user_query, context):
    # API Rate Limiting
    time.sleep(5)
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # # system message
    # system_content = (
    #     "<コンテキスト>のみを参考にして、質問に関連する情報をすべて取得してまとめてください。\n\n"
    #     "<注意点>\n"
    #     "- コンテキストのみから情報を抽出してください。\n"
    #     "- コンテキストに答えの手がかりが見つからないと判断される場合は、「分かりません」と答えること。\n"
    # )

    # user message
    user_content = (
        "<コンテキスト>のみを参考にして、質問に関連する情報をすべて取得してまとめてください。\n\n"
        "<注意点>\n"
        "- コンテキストのみから情報を抽出してください。\n"
        "- コンテキストに答えの手がかりが見つからないと判断される場合は、「分かりません」と答えること。\n\n"
        "<質問>\n"
        f"{user_query}\n\n"
        "<コンテキスト>\n"
        f"{context}\n\n"
    )

    # messages = [
    #     # {"role": "system", "content": system_content}, # system message
    #     {"role": "user", "content": user_content}, # user message
    # ]

    # Generate response using Gemini model with custom parameters
    # response = model.generate_content(
    #     [user_content],
    #     max_tokens=30000,     # Maximum number of tokens in the response
    #     temperature=0,   # Adjust the temperature for randomness
    #     top_p=0.95,        # Nucleus sampling
    #     frequency_penalty=0,
    #     presence_penalty=0,
    # )
    response = model.generate_content([user_content])

    # print("\n===== 回答 =====\n")
    # print(response)
    return response.text.strip() if response and response.text else None


