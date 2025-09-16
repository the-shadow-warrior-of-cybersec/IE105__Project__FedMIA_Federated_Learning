# Đồ án IE105 – Tìm hiểu cách tấn công trong mô hình Học Liên kết

## 📌 Giới thiệu

Dự án này tập trung nghiên cứu và triển khai **Tấn công Suy luận Thành viên** (_Membership Inference Attack – MIA_) trong môi trường **Học Liên kết** (_Federated Learning – FL_), cụ thể là phương pháp **FedMIA**.  
Phương pháp này khai thác nguyên lý _“All for One”_ – sử dụng thông tin từ tất cả các client để tăng độ chính xác và hiệu quả của cuộc tấn công.

Mục tiêu của dự án:

- Hiểu rõ cơ chế hoạt động của FL và MIA.
    
- Thực hành triển khai FedMIA trên một hệ thống FL giả lập.
    
- Đánh giá và trực quan hóa kết quả tấn công.
    

## 🗂️ Cấu trúc Dự án

```
├── experiments/                     # Thư mục chứa script thực nghiệm & kết quả
├── models/                          # Định nghĩa mô hình học máy
├── utils/                           # Hàm tiện ích hỗ trợ
├── dataset.py                       # Xử lý & tải dữ liệu
├── _federated_learning.py           # Triển khai học liên kết (server & client)
├── _fed_membership_attack.py        # Triển khai tấn công FedMIA
├── 1_run_federated_learning.sh      # Script shell chạy học liên kết
├── 2_run_fed_membership_attack.sh   # Script shell chạy tấn công FedMIA
├── requirements.txt                 # Danh sách thư viện phụ thuộc
└── README.md                        # Tài liệu hướng dẫn (file này)
```

> **Tip:** Thư mục `log/` sẽ tự động được tạo để lưu kết quả huấn luyện và tấn công; thư mục `_result` chứa kết quả của tác giả tham khảo.

## ⚙️ Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# hoặc .\venv\Scripts\activate trên Windows
```

### 2. Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

> **Yêu cầu:** Python ≥ 3.8

## 🚀 Hướng dẫn Sử dụng

### 1. Huấn luyện mô hình Học Liên kết

```bash
bash 1_run_federated_learning.sh
```

Sau khi chạy xong, mô hình FL và log sẽ nằm trong thư mục `log/`.

### 2. Thực hiện Tấn công FedMIA

```bash
bash 2_run_fed_membership_attack.sh
```

Kết quả tấn công (độ chính xác, biểu đồ…) sẽ được lưu trong `log/` và `_result`.

## 🧪 Kết quả

- **Đầu ra huấn luyện:** mô hình FL đã huấn luyện, log loss/accuracy theo vòng lặp.
    
- **Đầu ra tấn công:** xác suất thành viên, độ chính xác của FedMIA, biểu đồ trực quan hóa.
    
- **So sánh:** người dùng có thể đối chiếu với kết quả mẫu trong `_result` để kiểm chứng.
    

## 📝 Đóng góp

Nếu muốn đóng góp (ví dụ thêm mô hình mới, cải thiện FedMIA, hoặc bổ sung dataset), bạn có thể fork dự án và tạo pull request.

## 📚 Tài liệu Tham khảo

- **Paper gốc:** FedMIA: _An Effective Membership Inference Attack Exploiting the “All for One” Principle in Federated Learning_.
    
- **Mã nguồn tham khảo:** [Liar-Mask/FedMIA](https://github.com/Liar-Mask/FedMIA)
    

---
