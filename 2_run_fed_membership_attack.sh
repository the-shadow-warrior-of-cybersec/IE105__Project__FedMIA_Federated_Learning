# Thiết lập các siêu tham số và cấu hình
path="log_fedmia/iid"
seed=2025
total_epoch=25
gpu=0

# Chạy tấn công
python -u _fed_membership_attack.py  ${path} ${total_epoch} ${gpu} ${seed}  