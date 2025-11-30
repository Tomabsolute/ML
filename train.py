from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
while True:
    prompt = input("请输入你的问题(exit退出)：")
    if prompt == "exit":
        break
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 返回 input_ids + attention_mask
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    generated_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=512
    )

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)