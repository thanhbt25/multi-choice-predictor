import os
import json
import time
import csv
import torch
from collections import Counter, defaultdict

import config
import utils
import models

def main():
    question_times = defaultdict(float)

    if not os.path.exists(config.INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file: {config.INPUT_FILE}")

    with open(config.INPUT_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # PASS 1: RUN ORIGINAL VARIANT
    # ----------------------------------------------------------
    pass1_inputs = []
    all_variants_dict = {}

    for s_idx, sample in enumerate(test_data):
        choices = sample["choices"]
        variants = utils.get_shuffled_variants(choices, num_variants=config.NUM_TTA_RUNS)
        all_variants_dict[s_idx] = variants
        
        var_0 = variants[0]
        pass1_inputs.append({
            "sample_idx": s_idx,
            "variant_idx": 0,
            "question": sample["question"],
            "choices": var_0["choices"],
            "mapping": var_0["mapping"]
        })

    pass1_results = []
    batches_pass1 = [pass1_inputs[i:i + config.BATCH_SIZE] for i in range(0, len(pass1_inputs), config.BATCH_SIZE)]

    for batch in batches_pass1:
        batch_start_time = time.time()
        results = models.dispatch_predict_batch(batch)
        batch_elapsed = time.time() - batch_start_time
        
        per_sample_time = batch_elapsed / len(batch)
        
        for t_input, res in zip(batch, results):
            s_idx = t_input["sample_idx"]
            question_times[s_idx] += per_sample_time
            
            mapped_pred = t_input["mapping"][res["pred"]]
            pass1_results.append({
                "sample_idx": s_idx,
                "mapped_pred": mapped_pred,
                "confidence": res["confidence"]
            })

    # ----------------------------------------------------------
    # CLASSIFY PASS 1 RESULTS & TRIGGER TTA IF NEEDED
    # ----------------------------------------------------------
    final_predictions = {}
    pass2_inputs = []
    uncertain_results = defaultdict(list)

    for res in pass1_results:
        s_idx = res["sample_idx"]
        if res["confidence"] >= config.CONFIDENCE_THRESHOLD:
            final_predictions[s_idx] = res["mapped_pred"]
        else:
            uncertain_results[s_idx].append(res["mapped_pred"])
            
            variants = all_variants_dict[s_idx]
            for v_idx in range(1, len(variants)):
                var = variants[v_idx]
                sample = test_data[s_idx]
                pass2_inputs.append({
                    "sample_idx": s_idx,
                    "variant_idx": v_idx,
                    "question": sample["question"],
                    "choices": var["choices"],
                    "mapping": var["mapping"]
                })

    # ----------------------------------------------------------
    # PASS 2: CHẠY TTA CHO CÁC CÂU CÓ CONFIDENCE THẤP
    # ----------------------------------------------------------
    if pass2_inputs:
        batches_pass2 = [pass2_inputs[i:i + config.BATCH_SIZE] for i in range(0, len(pass2_inputs), config.BATCH_SIZE)]
        
        for batch in batches_pass2:
            batch_start_time = time.time()
            results = models.dispatch_predict_batch(batch)
            batch_elapsed = time.time() - batch_start_time
            
            per_sample_time = batch_elapsed / len(batch)
            
            for t_input, res in zip(batch, results):
                s_idx = t_input["sample_idx"]
                question_times[s_idx] += per_sample_time
                
                mapped_pred = t_input["mapping"][res["pred"]]
                uncertain_results[s_idx].append(mapped_pred)

        for s_idx, pred_list in uncertain_results.items():
            votes = Counter(pred_list)
            # Lấy label có số lượng vote lớn nhất
            final_pred = votes.most_common(1)[0][0]
            final_predictions[s_idx] = final_pred

    # ----------------------------------------------------------
    # EXPORT SUBMISSION CSV FILES
    # ----------------------------------------------------------
    # Đọc nhanh danh sách qid theo đúng thứ tự câu hỏi ban đầu
    qids = [sample.get("qid", f"unknown_{s_idx}") for s_idx, sample in enumerate(test_data)]

    # 1. Ghi file submission.csv (qid,answer)
    with open("code/submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer"])
        for s_idx, qid in enumerate(qids):
            writer.writerow([qid, final_predictions[s_idx]])

    # 2. Ghi file submission_time.csv (qid,answer,time)
    with open("code/submission_time.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "time"])
        for s_idx, qid in enumerate(qids):
            q_time = question_times[s_idx]
            writer.writerow([qid, final_predictions[s_idx], f"{q_time:.4f}"])

if __name__ == "__main__":
    main()