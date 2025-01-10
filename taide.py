
# 逐步生成函數
""" from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessorList
import torch
import time

# 設置量化配置
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1")
model = AutoModelForCausalLM.from_pretrained(
    "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
    quantization_config=quantization_config,
    device_map="auto"
)

# 逐步生成函數
def incremental_generate(model, tokenizer, inputs, max_new_tokens=1, stop_token=None, delay=0.05):

    generated_ids = inputs["input_ids"].clone()
    for _ in range(2000):  # 最大生成限制
        outputs = model.generate(
            generated_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList()
        )
        
        new_ids = outputs[0][generated_ids.shape[-1]:]  # 獲取新生成的 token
        new_ids = new_ids.unsqueeze(0)
        generated_ids = torch.cat([generated_ids, new_ids], dim=-1)
        
        # 解碼新生成的 token 並逐字顯示
        new_text = tokenizer.decode(new_ids.squeeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for char in new_text:
            print(char, end="", flush=True)
            time.sleep(delay)
        
        # 停止條件
        if stop_token and stop_token in new_text:
            break

    print()  # 打印換行

# 主交互迴圈
while True:
    input_text = input("User: ")
    if input_text.lower() == "exit":
        break

    message = [
        {"role": "system", "content": "你是一個有用的助手"},
        {"role": "user", "content": input_text}
    ]
    # 構造輸入文本
    text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 逐步生成輸出
    print("TAIDE: ", end="")
    incremental_generate(model, tokenizer, model_inputs, max_new_tokens=1, stop_token=None)
 """
############################################################################################################
# 正常一次性輸出
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1")
model = AutoModelForCausalLM.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
                                             quantization_config=quantization_config,
                                             device_map = "auto")

while True:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_text = input("User: ")
    if input_text == "exit":
        break
    message = [
        {"role": "system", "content": "你是一個有用的助手"},
        {"role": "user", "content": input_text}
    ]
    text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs  = model.generate(**model_inputs, max_new_tokens=2000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in reply:
        reply = reply.split("assistant")[1].strip()
    else:
        reply = reply.strip()


    print(f"TAIDE: {reply.strip()}\n")