from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.memory import ConversationBufferMemory  
from llama_index.embeddings.openai import OpenAIEmbedding
import re
import os
import torch
from dotenv import load_dotenv 

# 初始化 Flask 應用和 CORS 支持
app = Flask(__name__, static_folder="../GUI")
CORS(app)
# 定義 Flask API 路由
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)
# 加載環境變量
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 初始化 TAIDE 模型
quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1")
model = AutoModelForCausalLM.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
                                             quantization_config=quantization_config,
                                             device_map = "auto")

# 全局會話記憶字典，用於多用戶會話管理
conversation_memories = {}

# 清理文本的輔助函數
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # 去除多餘空格
    text = re.sub(r'[^\w\s.,]', '', text)  # 去除特殊字符
    return text

# 加載和切割 PDF 文檔
def load_pdf_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 合併文檔文本並清理
    full_text = " ".join([doc.page_content for doc in docs])
    full_text = clean_text(full_text)
    print(f"合併後內容總長度: {len(full_text)}")

    # 使用更優化的切割策略
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,  # 每塊最大字符數
        chunk_overlap=500  # 重疊字符數
    )
    chunks = text_splitter.split_text(full_text)
    print(f"分割後的文檔塊數量: {len(chunks)}")

    # 將文本塊轉換為 Document 對象
    documents = [Document(text=chunk) for chunk in chunks]
    return documents

# 使用TAIDE回應
def generate_response_with_taide(input_text, context, memory_context):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    message = [
        {"role": "system", "content": "你是一個有用的助手"},
        {"role": "user", "content": f"""以下內容為搜索到的東西：\n\n{context}\n\n使用者問題：{input_text}，請根據使用者的問題去搜索到的內容找答案，
                                    你也可以加入你的想法，若根據搜索到的內容可以找出答案，請直接回答問題；
                                    若你認為使用者問題不明確，請輸出一點提示，幫助使用者收斂問題；
                                    若有需要，你可以參考之前對話記憶：\n\n{memory_context}
                                    ，並給出信心分數[1]~[10]，給出"[10]"代表系統越信心。"""}
    ]
    text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors="pt").to(device)
    print('1')
    outputs  = model.generate(**model_inputs, max_new_tokens=2000, num_return_sequences=1, do_sample=True,
                              pad_token_id=tokenizer.eos_token_id, top_k=50, top_p=0.95)
    print('2')
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in reply:
        reply = reply.split("assistant")[1].strip()

    return reply


# 加載或重新建立向量索引
pdf_file_path = "programe/document/GSAF21091303_Service_Manual_of_HW3.2_for_TMAA_and_TMAAM_TW.pdf"  
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
save_dir = "C:/Users/606/Desktop/TM_project/save_index"

if os.path.exists(save_dir):
    print("使用先前資料...........")
    storage_context = StorageContext.from_defaults(persist_dir=save_dir)
    index = load_index_from_storage(storage_context)
else:
    print("重新建立向量...........")
    os.makedirs(save_dir, exist_ok=True)
    pdf_documents = load_pdf_document(pdf_file_path)
    index = VectorStoreIndex.from_documents(pdf_documents, embeddings=embedding_model)
    index.storage_context.persist(persist_dir=save_dir)



@app.route('/api/get-response', methods=['POST'])
def ask_question():
    # 解析用戶的請求
    data = request.json
    input_text = data.get("question", "")
    conversation_id = data.get("conversationId", "default")
    print('-' * 50)
    print("conversation_id: ", conversation_id)

    # 確保每個用戶有獨立的記憶模組
    if conversation_id not in conversation_memories:
        conversation_memories[conversation_id] = ConversationBufferMemory()  # 每個用戶一個記憶體

    memory = conversation_memories[conversation_id]

    ################ 檢索相關文檔 ################
    retriever = index.as_retriever(top_k=10)
    retrieved_docs = retriever.retrieve(input_text)
    print('*' * 100)
    print(f"返回的文檔數量: {len(retrieved_docs)}")

    if retrieved_docs:
        context = "\n".join([doc.text for doc in retrieved_docs])
        # 輸出檢索到的文檔
        print("context: ", context)
        print('-' * 100)

        # 獲取記憶上下文並構造 LLM 請求
        memory_context = memory.load_memory_variables({}).get("history", "")
        print("TAIDE 生成回復中...........")
        response = generate_response_with_taide(input_text, context, memory_context)
        print("TAIDE 回復完成。")

        # 更新記憶
        memory.save_context({"input": input_text}, {"output": response})
        print("記憶已更新。")
        print("memory: ", memory_context)
        print('*'*100)
    else:
        # 若無檢索結果，生成澄清性問題
        clarifying_question = f"您的問題過於模糊，請提供更多具體信息。"
        memory.save_context({"input": input_text}, {"output": clarifying_question})
        return jsonify({"answer": clarifying_question})

    # 返回答案
    formatted_response = response.replace('\n', '\\n')
    return jsonify({'answer': formatted_response})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
