clear

echo "#################### Note ####################"
echo
echo "- CIFAR-100 (cifar100) là một bộ dữ liệu hình ảnh phổ biến được sử dụng để huấn luyện và đánh giá các mô hình học sâu, đặc biệt là trong các bài toán phân loại ảnh nhiều lớp"
echo
echo "- CICMalDroid 2020 (cicmaldroid) là một bộ dữ liệu phần mềm độc hại Android được phát triển bởi CIC và CIRA, nhằm hỗ trợ nghiên cứu trong lĩnh vực phân loại và phát hiện phần mềm độc hại trên nền tảng AndroidDataset."
echo 

echo "#################### Thiết lập các siêu tham số và cấu hình (có thể tùy biến thêm trong file) ####################"
echo

# Nhập tên dataset và kiểm tra
read -p "Nhập tên dataset (cifar100/cicmaldroid): " dataset
if [[ "$dataset" != "cifar100" && "$dataset" != "cicmaldroid" ]]; then
    echo "Lỗi: Dataset phải là cifar100 hoặc cicmaldroid."
    exit 1
fi
echo

if [ "$dataset" == "cifar100" ]; then
    model_name="alexnet"
    echo "---> Tự chọn model phù hợp: AlexNet"
else
    model_name="mlp"
    echo "---> Tự chọn model phù hợp: Multilayer Perceptrontron"
fi

echo
echo "---> Tự chọn optimizer phù hợp: Stochastic Gradient Descent"
echo
opt=sgd

# Kiểm tra seed là số nguyên
read -p "Nhập seed (int): " seed
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Lỗi: Seed phải là số nguyên."
    exit 1
fi
echo

# Kiểm tra learning rate là số (float hoặc int)
read -p "Nhập learning rate (nên nhỏ hơn hoặc bằng 0.1): " lr
if ! [[ "$lr" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Lỗi: Learning rate phải là số."
    exit 1
fi
echo

# Nhập tổng số epoch
read -p "Nhập tổng số epoch: " epochs
if ! [[ "$epochs" =~ ^[0-9]+$ ]]; then
    echo "Lỗi: epochs phải là số nguyên."
    exit 1
fi
echo

# Kiểm tra local_epoch là số nguyên
read -p "Nhập số epoch client train trước khi gửi lên server: " local_epoch
if ! [[ "$local_epoch" =~ ^[0-9]+$ ]]; then
    echo "Lỗi: local_epoch phải là số nguyên."
    exit 1
fi
echo

# In menu
echo "#################### Chọn chế độ thử nghiệm (cách chia dataset cho các client) ####################"
echo
echo "1. Chế độ IID (dữ liệu ở các client có nhiều tương quan)"
echo
echo "2. Chế độ non-IID (dữ liệu ở các client phân biệt)" 
echo
read -p "---> Nhập lựa chọn của bạn (1/2): " choice
echo


# Chạy thử nghiệm dựa trên lựa chọn của người dùng
if [ "$choice" = "1" ]; then
    # iid experiment
    save_dir=log_fedmia/iid
    mkdir -p "$save_dir"
    echo "#################### Thử nghiệm IID ####################"
    echo
    CUDA_VISIBLE_DEVICES=0 python _federated_learning.py --seed $seed --num_users 10 --iid 1 \
     --dataset $dataset --model_name $model_name --epochs $epochs --local_ep $local_epoch \
     --lr $lr --batch_size 64 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
     --lr_up cosine --MIA_mode 1 --gpu 0 2>&1 | tee "${save_dir}/raw_logs"
elif [ "$choice" = "2" ]; then
    # non-iid experiment
    save_dir=log_fedmia/noniid
    mkdir -p "$save_dir"
    echo "#################### Thử nghiệm Non-IID ####################"
    echo
    CUDA_VISIBLE_DEVICES=0 python _federated_learning.py --seed $seed --num_users 10 --iid 0 --beta 1.0 \
     --dataset $dataset --model_name $model_name --epochs $epochs --local_ep $local_epoch \
     --lr $lr --batch_size 64 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
     --lr_up cosine --MIA_mode 1 --gpu 0 2>&1 | tee "${save_dir}/raw_logs"
else
    echo "Invalid choice. Please run the script again and select 1 or 2."
    exit 1
fi

echo
echo "#################### Hoàn thành thử nghiệm ####################"
echo
