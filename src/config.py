import os
import re

# Thiết lập quản lý bộ nhớ VRAM cho PyTorch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Cấu hình Model & Dữ liệu
MODEL_NAME   = "Qwen/Qwen3.5-4B"
INPUT_FILE   = "../code/public-test_1780368312.json" 

# Tham số Inference
BATCH_SIZE   = 8
USE_COMPILE  = False

GAP_THRESHOLDS = {
    "high_gap": 0.50,
    "med_gap": 0.20
}
NUM_TTA_RUNS = 3
CONFIDENCE_THRESHOLD = 0.70
MAX_NEW_TOKEN = 150 

# Các Pattern ký tự Toán học / Biểu thức
MATH_SYMBOL_PATTERNS = [
    r"\\frac", r"\\dfrac", r"\\binom", r"\\sum", r"\\prod", r"\\int", r"\\lim", 
    r"\\partial", r"\\sin", r"\\cos", r"\\tan", r"\\log", r"\\ln", r"\\exp",
    r"\\le", r"\\ge", r"\\ll", r"\\gg", r"\\therefore", r"\\because", r"\\pi", 
    r"\\pm", r"\\propto", r"\\approx", r"\\cdot", r"\\div", 
    r"\\begin\{cases\}", r"\\begin\{matrix\}", r"\\left", r"\\right",
    r"\+", r"-", r"\*", r"/", r"\^", r"\*\*", r"sqrt", r"root", r"%", r"mod", r"abs",
    r"\|.*\|", r"=", r"≠", r"≈", r"<", r">", r"≤", r"≥", r"∈", r"∉", r"⊂", r"⊆", 
    r"→", r"->", r"⇒", r"=>", r"⇔", r"<=>", r"π", r"e", r"∞"
]
COMBINED_MATH_REGEX = re.compile("|".join(MATH_SYMBOL_PATTERNS))

# Từ khóa định vị câu hỏi cần suy luận CoT
REASONING_KEYWORDS = {
    "tính tỷ lệ", "tính phần trăm", "tính tăng trưởng", "tính lãi suất", "tính thể tích", 
    "tính diện tích", "đổi đơn vị", "tính khấu hao", "định khoản kế toán", "xác định tài khoản", 
    "tính thuế gtgt", "ttđb", "tndn", "thuế nhập khẩu", "tính tỷ giá", "cán cân thương mại", 
    "xuất khẩu ròng", "tính sản lượng tối đa", "doanh thu tối đa", "lợi nhuận tối đa", 
    "tính chi phí biên", "doanh thu biên", "co giãn cầu", "giải phương trình", "giải hệ phương trình", 
    "tìm nghiệm", "biểu diễn đồ thị", "tính tích phân", "tính đạo hàm", "tính giới hạn", 
    "tìm cực trị", "tính xác suất", "phân phối chuẩn", "kỳ vọng", "phương sai", "ước lượng", 
    "tính đạo hàm riêng", "tối ưu hóa ràng buộc", "tính xác suất có điều kiện", "biến cố đối lập", 
    "công thức bayes", "tính khoảng cách", "tọa độ", "vector", "tính điểm trung bình", "xếp loại",
    "tính nồng độ", "tính ph", "tính tích số tan", "tính hằng số cân bằng", "tính hiệu ứng nhiệt", 
    "ΔH", "ΔS", "ΔG", "tính số lượng chất", "hiệu suất phản ứng", "tính số bit", "địa chỉ bộ nhớ", 
    "chuyển đổi cơ số", "tính tần số", "công suất", "điện trở", "năng lượng", "tính số nguyên tử", 
    "số nuclôn", "độ hụt khối", "tính nhiệt độ sôi", "tính tan", "tính mức phạt", "thời hiệu", 
    "thời hạn theo luật", "tính ngạch", "bậc lương", "phụ cấp", "tính số lượng đại biểu", "tỷ lệ dân số",
    "chứng minh", "suy luận", "suy ra", "giả sử", "phân tích", "so sánh", "ước lượng", "làm tròn", 
    "xấp xỉ", "kiểm tra điều kiện"
}