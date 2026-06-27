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
    # Khởi tạo dictionary theo dõi mốc thời gian thực tế của từng câu hỏi độc lập
    question_start_times = {}
    question_times = defaultdict(float)

    if not os.path.exists(config.INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình đầu vào: {config.INPUT_FILE}")

    with open(config.INPUT_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Giải phóng bộ nhớ đệm VRAM trước khi bắt đầu pipeline suy luận
    torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # PASS 1: RUN ORIGINAL VARIANT (XỬ LÝ LUỒNG BIẾN THỂ GỐC)
    # ----------------------------------------------------------
    pass1_inputs = []
    all_variants_dict = {}

    for s_idx, sample in enumerate(test_data):
        # ⏱️ MỐC T1: Đánh dấu thời điểm câu hỏi bắt đầu được đưa vào pipeline xử lý
        question_start_times[s_idx] = time.time()
        
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

    # Thực thi inference Pass 1 qua cơ chế Batching để tối ưu hóa hiệu năng GPU
    for batch in batches_pass1:
        results = models.dispatch_predict_batch(batch)
        
        for t_input, res in zip(batch, results):
            s_idx = t_input["sample_idx"]
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
        
        # Nếu mô hình có độ tự tin cao vượt ngưỡng threshold -> Chốt đáp án ngay
        if res["confidence"] >= config.CONFIDENCE_THRESHOLD:
            final_predictions[s_idx] = res["mapped_pred"]
            
            # ⏱️ MỐC T2-A: Câu hỏi kết thúc sớm tại Pass 1 -> Tính thời gian xử lý thực tế
            question_times[s_idx] = time.time() - question_start_times[s_idx]
        else:
            # Ngược lại, nếu độ tự tin thấp -> Đưa câu hỏi vào hàng đợi Pass 2 để kích hoạt TTA
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
    # PASS 2: CHẠY TTA (TEST-TIME AUGMENTATION) CHO CÁC CÂU CÓ CONFIDENCE THẤP
    # ----------------------------------------------------------
    if pass2_inputs:
        batches_pass2 = [pass2_inputs[i:i + config.BATCH_SIZE] for i in range(0, len(pass2_inputs), config.BATCH_SIZE)]
        
        for batch in batches_pass2:
            batch_start_time = time.time()
            results = models.dispatch_predict_batch(batch)
            batch_elapsed = time.time() - batch_start_time
            
            # Phân bổ thời gian thực thi của batch một cách đồng đều cho các biến thể trong batch đó
            per_variant_time = batch_elapsed / len(batch)
            
            for t_input, res in zip(batch, results):
                s_idx = t_input["sample_idx"]
                
                # Tích lũy thời gian xử lý của riêng các biến thể TTA vào câu hỏi tương ứng
                question_times[s_idx] += per_variant_time
                
                mapped_pred = t_input["mapping"][res["pred"]]
                uncertain_results[s_idx].append(mapped_pred)

        # Tiến hành bầu chọn đa số (Majority Voting) để đưa ra kết luận đáp án chuẩn xác nhất
        for s_idx, pred_list in uncertain_results.items():
            votes = Counter(pred_list)
            final_pred = votes.most_common(1)[0][0]
            final_predictions[s_idx] = final_pred
            
            # ⏱️ MỐC T2-B: Tính toán tổng thời gian hoàn chỉnh cho câu hỏi phải trải qua TTA
            # Tổng thời gian = (Thời gian xử lý nền ở Pass 1) + (Tổng thời gian của các biến thể tại Pass 2)
            time_spent_in_pass1 = batch_start_time - question_start_times[s_idx]
            question_times[s_idx] += time_spent_in_pass1

    # ----------------------------------------------------------
    # EXPORT SUBMISSION CSV FILES (XUẤT FILE KẾT QUẢ THEO CHUẨN BTC)
    # ----------------------------------------------------------
    # Lấy thông tin mã định danh câu hỏi (qid) đảm bảo đúng thứ tự gốc
    qids = [sample.get("qid", f"unknown_{s_idx}") for s_idx, sample in enumerate(test_data)]

    # Đảm bảo thư mục đầu ra tồn tại trước khi tiến hành ghi dữ liệu
    os.makedirs("code", exist_ok=True)

    # 1. Ghi và xuất file kết quả dự đoán đáp án: submission.csv
    with open("code/submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer"])
        for s_idx, qid in enumerate(qids):
            writer.writerow([qid, final_predictions[s_idx]])

    # 2. Ghi và xuất file thống kê thời gian thực thi: submission_time.csv
    with open("code/submission_time.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "time"])
        for s_idx, qid in enumerate(qids):
            q_time = question_times[s_idx]
            writer.writerow([qid, final_predictions[s_idx], f"{q_time:.4f}"])

    print("HOÀN THÀNH. Đã lưu file CSV theo chuẩn BTC tại thư mục /code.")

if __name__ == "__main__":
    main()