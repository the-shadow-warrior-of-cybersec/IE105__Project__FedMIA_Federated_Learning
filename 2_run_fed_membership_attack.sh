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

# Kiểm tra xem thư mục đã tồn tại chưa
if [ ! -d "$path" ]; then
    echo "Lỗi: Chưa có log về việc trên dữ liệu $temp!"
    echo
    exit 1
fi

read -p "Nhập số tổng epoch được dùng để train: " total_epoch
if ! [[ "$total_epoch" =~ ^[0-9]+$ ]]; then
    echo "Lỗi: Total epoch phải là số nguyên!"
    exit 1
fi
echo

read -p "Nhập chế độ tấn công (train/val/test/mix): " attack_mode
if [[ "$attack_mode" != "train" && "$attack_mode" != "val" && "$attack_mode" != "test" && "$attack_mode" != "mix" ]]; then
    echo "Lỗi: Chế độ tấn công không tồn tại!"
    exit 1
fi
echo

read -p "Nhập seed (int): " seed
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Lỗi: Seed phải là số nguyên!"
    exit 1
fi
echo

echo "#################### Quy trình tấn công ####################"
echo

gpu_index=0 # Lấy index ứng với index GPU của máy
python -u _fed_membership_attack.py  ${path} ${total_epoch} ${attack_mode} ${gpu_index} ${seed}