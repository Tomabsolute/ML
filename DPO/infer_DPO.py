from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

# =============== 1. 路径配置 ===============
CKPT_DIR = "./Qwen2-0.5B-DPO-stable/checkpoint-62000"   # 换成你自己最后的 checkpoint
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
device = "cuda"

# =============== 2. 加载模型 ===============
model = AutoModelForCausalLM.from_pretrained(
    CKPT_DIR,
    dtype=torch.bfloat16,         # 和训练一致
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# pad_token 修正
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

while True:
    prompt = input("请输入你的问题：(exit退出)")
    if prompt == "exit":
        break
    # =============== 3. 构建输入 ===============
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # =============== 4. 正式生成 ===============
    gen_cfg = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_cfg,
        )

    # =============== 5. 解码输出 ===============
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print("\n=== Model Output ===\n")
    print(response)