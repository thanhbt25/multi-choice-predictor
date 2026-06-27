import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import utils

# print("=" * 60)
# print("LOADING FAST MODEL (FP16 - SINGLE GPU)")
# print("=" * 60)
load_start = time.time()

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda:0", 
    attn_implementation="sdpa",
)
model.eval()

if config.USE_COMPILE:
    # print("Compiling model…")
    model = torch.compile(model, mode="reduce-overhead")

# print(f"Model loaded in {time.time() - load_start:.2f}s")

# Cache token IDs cho các chữ cái nhãn lựa chọn
MAX_CHOICES = 26
LABEL_TOKEN_IDS = []
for i in range(MAX_CHOICES):
    letter = chr(65 + i)
    tok_ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    LABEL_TOKEN_IDS.append(tok_ids[0])

@torch.inference_mode()
def predict_batch_normal(batch: list[dict]) -> list[dict]:
    prompts = [utils.format_prompt(s["question"], s["choices"]) for s in batch]
    n_choices_per = [len(s["choices"]) for s in batch]

    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_logits = outputs.logits[:, -1, :] 

    results = []
    for b_idx, sample in enumerate(batch):
        n = n_choices_per[b_idx]
        labels = [chr(65 + i) for i in range(n)]
        label_token_ids = LABEL_TOKEN_IDS[:n]
        
        label_logits = last_logits[b_idx, label_token_ids]
        log_probs = torch.log_softmax(label_logits, dim=0).float()
        probs = torch.softmax(label_logits, dim=0).float()

        prob_values = probs.cpu().numpy().tolist()
        sorted_probs = sorted(prob_values, reverse=True)
        
        pred_idx = int(probs.argmax())
        results.append({
            "pred": labels[pred_idx],
            "confidence": float(sorted_probs[0]),
            "conf_gap": (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0,
            "scores": {labels[i]: float(log_probs[i]) for i in range(n)},
            "prompt": prompts[b_idx]
        })
    return results

@torch.inference_mode()
def predict_batch_cot(batch: list[dict]) -> list[dict]:
    prompts = [utils.format_cot_prompt(s["question"], s["choices"]) for s in batch]
    n_choices_per = [len(s["choices"]) for s in batch]

    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)

    stop_words = ["user", "User:", "\nuser", "\nUser"]
    bad_words_ids = [tokenizer.encode("<think>", add_special_tokens=False)]
    
    gen_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=config.MAX_NEW_TOKEN,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bad_words_ids=bad_words_ids,
        repetition_penalty=1.1
    )

    results = []
    for b_idx, sample in enumerate(batch):
        n = n_choices_per[b_idx]
        labels = [chr(65 + i) for i in range(n)]
        
        prompt_len = input_ids[b_idx].shape[0]
        decoded_output = tokenizer.decode(gen_outputs[b_idx][prompt_len:], skip_special_tokens=True)
        
        clean_output = decoded_output
        for stop_w in stop_words:
            if stop_w in clean_output:
                clean_output = clean_output.split(stop_w)[0]
        
        clean_output = clean_output.strip()
        pred = None
        upper_output = clean_output.upper()
        
        last_keyword_idx = upper_output.rfind("ĐÁP ÁN")
        if last_keyword_idx != -1:
            after_keyword = upper_output[last_keyword_idx:]
            for char in after_keyword:
                if char in labels:
                    pred = char
                    break
                    
        if not pred:
            for char in reversed(upper_output):
                if char in labels:
                    pred = char
                    break

        if not pred: 
            pred = labels[0]

        # if b_idx == 0:
        #     print("\n" + "-"*40)
        #     print(f"📝 [DEBUG CoT] {sample['question'][:60]}...")
        #     print(f"🤖 [OUTPUT]: {clean_output}")
        #     print(f"🎯 [PRED]: {pred}")
        #     print("-"*40)

        results.append({
            "pred": pred,
            "confidence": 1.0, 
            "conf_gap": 1.0,
            "scores": {l: (0.0 if l == pred else -float('inf')) for l in labels},
            "prompt": prompts[b_idx]
        })
    return results

def dispatch_predict_batch(batch: list[dict]) -> list[dict]:
    normal_samples = []
    cot_samples = []
    
    for s in batch:
        if utils.is_reasoning_question(s["question"], s["choices"]):
            cot_samples.append(s)
        else:
            normal_samples.append(s)
            
    res_map = {}
    if normal_samples:
        normal_res = predict_batch_normal(normal_samples)
        for s, r in zip(normal_samples, normal_res):
            res_map[id(s)] = r
            
    if cot_samples:
        cot_res = predict_batch_cot(cot_samples)
        for s, r in zip(cot_samples, cot_res):
            res_map[id(s)] = r
            
    return [res_map[id(s)] for s in batch]