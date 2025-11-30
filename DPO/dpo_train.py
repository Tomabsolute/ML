from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# 1. 模型：用 bfloat16（4090 支持），比 fp16 稳定很多
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,   # 关键：改成 bf16，而不是 fp16
    device_map="auto",
)

# 根据显存情况，先不开梯度检查点；如果再 OOM，再加
model.config.use_cache = False

# 2. tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 3. 数据集
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 4. DPOConfig：在官方基础上只加“必要的”几个参数
training_args = DPOConfig(
    output_dir="Qwen2-0.5B-DPO-stable",

    # 和官方一样：batch 很小，避免显存炸
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,

    # 序列长度（如果 OOM，可以再减）
    max_length=1024,
    max_prompt_length=512,

    # 关键：用 bf16 而不是 fp16
    bf16=True,                    # 让 accelerate 用 bf16 训练
    gradient_checkpointing=False, # 先关，OOM 再开
    remove_unused_columns=False,

    # 关键：防炸配置
    learning_rate=2e-6,           # 比 5e-6 更稳
    max_grad_norm=0.5,            # 防梯度爆炸，非常重要

    # 其他日志和保存设置
    num_train_epochs=1,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

if __name__ == "__main__":
    trainer.train()