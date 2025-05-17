# Thiết lập các siêu tham số và cấu hình
dataset=cicmaldroid # cifar100/ dermnet/ cicmaldroid
model_name=mlp # alexnet/ ResNet18/ mlp
opt=sgd
seed=2025
lr=0.1 # 0.1 for alexnet; 0.01 for mlp
local_epoch=1


# In menu
clear
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
     --dataset $dataset --model_name $model_name --epochs 25 --local_ep $local_epoch \
     --lr $lr --batch_size 64 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
     --lr_up cosine --MIA_mode 1 --gpu 0 2>&1 | tee "${save_dir}/raw_logs"
elif [ "$choice" = "2" ]; then
    # non-iid experiment
    save_dir=log_fedmia/noniid
    mkdir -p "$save_dir"
    echo "#################### Thử nghiệm Non-IID ####################"
    echo
    CUDA_VISIBLE_DEVICES=0 python _federated_learning.py --seed $seed --num_users 10 --iid 0 --beta 1.0 \
     --dataset $dataset --model_name $model_name --epochs 25 --local_ep $local_epoch \
     --lr $lr --batch_size 64 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
     --lr_up cosine --MIA_mode 1 --gpu 0 2>&1 | tee "${save_dir}/raw_logs"
else
    echo "Invalid choice. Please run the script again and select 1 or 2."
    exit 1
fi

echo
echo "#################### Hoàn thành thử nghiệm ####################"