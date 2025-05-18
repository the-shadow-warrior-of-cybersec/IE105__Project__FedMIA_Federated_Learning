clear
echo "#################### Thiết lập các siêu tham số và cấu hình ####################"
echo

read -p "Chọn cách chia dữ liệu đã được dùng (iid/noniid): " temp
if [[ "$temp" != "iid" && "$temp" != "noniid" ]]; then
    echo "Lỗi: Cách chia dữ liệu phải là iid hoặc noniid!"
    exit 1
fi

# Lấy thư mục lưu trữ tài nguyên phục vụ tấn công
path="log_fedmia/$temp"
echo

read -p "Nhập số tổng epoch được dùng để train: " total_epoch
if ! [[ "$total_epoch" =~ ^[0-9]+$ ]]; then
    echo "Lỗi: Total epoch phải là số nguyên!"
    exit 1
fi
echo

read -p "Nhập seed (int): " seed
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Lỗi: Seed phải là số nguyên!"
    exit 1
fi
echo

echo "#################### Attacking... ####################"
echo

# Lấy index ứng với index GPU của máy
gpu=0

# Chạy tấn công
python -u _fed_membership_attack.py  ${path} ${total_epoch} ${gpu} ${seed}  