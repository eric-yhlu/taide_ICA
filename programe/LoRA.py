import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

with open("document.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

# 處理巢狀結構
processed_data = []
for item in train_data:
    processed_item = {}
    for key, value in item.items():
        if isinstance(value, (dict, list)):  # 將巢狀結構轉為字串
            processed_item[key] = json.dumps(value, ensure_ascii=False)
        else:
            processed_item[key] = value
    processed_data.append(processed_item)

# 建立資料集
dataset = Dataset.from_list(processed_data)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print("Training dataset:",train_dataset)
print("Test dataset:",test_dataset)


# 載入模型
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 配置LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 資料處理
def preprocess_function(examples):
    inputs = examples["question"]
    targets = examples["answer"]

    # 批量處理，確保 input_ids 和 labels 是列表
    input_ids_batch = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )["input_ids"]
    target_ids_batch = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )["input_ids"]

    full_input_ids = []
    full_labels = []

    for input_ids, target_ids in zip(input_ids_batch, target_ids_batch):
        # 拼接輸入和輸出
        full_input_ids.append(input_ids + target_ids)
        # 創建標籤，輸入部分設為 -100
        full_labels.append([-100] * len(input_ids) + target_ids)

    return {"input_ids": full_input_ids, "labels": full_labels}


train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# train

training_args = TrainingArguments(
    output_dir="./output",
    learning_rate=5e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    overwrite_output_dir=True,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./lora_model")
# 評估模型
results = trainer.evaluate()
print(results)
