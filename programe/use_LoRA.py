from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# 載入微調後的模型和 tokenizer
model_name = "./lora_model"  # 微調後模型的存儲路徑
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def ask_question(question, model, tokenizer, max_length=128, temperature=0.7):
    """
    用於生成回答的推理函數
    :param question: 問題 (str)
    :param model: 微調後的模型
    :param tokenizer: 與模型對應的 tokenizer
    :param max_length: 生成回答的最大長度
    :param temperature: 控制生成的隨機性
    :return: 回答 (str)
    """
    device = next(model.parameters()).device
    # 將問題 tokenized
    inputs = tokenizer(question, return_tensors="pt", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} 
    # 使用模型生成輸出
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        top_p=0.95,  # 採樣方法，可選參數
        do_sample=True,  # 開啟隨機生成
        eos_token_id=tokenizer.eos_token_id,  # 停止標記
    )
    
    # 解碼模型生成的輸出
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

while True:
    # 問題輸入
    question = input("請輸入您的問題：")
    
    # 呼叫推理函數
    answer = ask_question(question, model, tokenizer)
    
    print(f"回答: {answer}")
    print('-'*50)

