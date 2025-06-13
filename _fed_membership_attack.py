import sys
import os
import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import scipy
import json
import warnings

# Loại bỏ các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# ==================== Chỉnh sửa hàm print để thực hiện tác vụ cụ thể ====================

print_to_console = print 
def print_to_file(*arg, end = None):
    global SAVE_DIR
    file_path = SAVE_DIR + f'/attack_select_{select_mode}_{select_method}20_{MODE}_n{SHADOW_NUM}_s{SEED}_running.log'
    if end == None:
        print_to_console(*arg, file=open(file_path, "a", encoding="utf-8"))
    else:
        print_to_console(*arg, end='', file=open(file_path, "a", encoding="utf-8"))
def print_to_everything(*arg, end = None):
    global SAVE_DIR
    file_path = SAVE_DIR + f'/attack_select_{select_mode}_{select_method}20_{MODE}_n{SHADOW_NUM}_s{SEED}_running.log'
    if end == None:
        print_to_console(*arg, file=open(file_path, "a", encoding="utf-8"))
    else:
        print_to_console(*arg, end='', file=open(file_path, "a", encoding="utf-8"))
    print_to_console(*arg, end=end)

# ==================== Tính toán các thông số cấn thiết cho thử nghiệm ====================

def liratio(mu_in, mu_out, var_in, var_out, new_samples):
    '''
    Tính giá trị phân phối tích lũy (CDF) của các mẫu mới dựa trên một phân phối chuẩn với trung bình (mu_out) và phương sai (var_out).
    '''
    l_out = scipy.stats.norm.cdf(new_samples, mu_out, np.sqrt(var_out))
    return l_out

@ torch.no_grad()
def hinge_loss_fn(x, y):
    '''
    Tính toán giá trị hàm mất mát hinge loss cho các mẫu đầu vào x và nhãn y.
    '''
    x, y = copy.deepcopy(x).cuda(),copy.deepcopy(y).cuda()
    mask = torch.eye(x.shape[1], device="cuda")[y].bool()
    tmp1 = x[mask]
    x[mask] =- 1e10
    tmp2 = torch.max(x,dim=1)[0]
    return (tmp1 - tmp2).cpu().numpy()

def ce_loss_fn(x, y):
    '''
    Tính mất mát cross entropy (CE) không gom gọn (reduction='none') giữa đầu ra (x) và nhãn (y).
    '''
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    return loss_fn(x,y)

def extract_hinge_loss(i):
    '''
    Tính toán giá trị hàm mất mát hinge loss cho các mẫu đầu vào.
    '''
    val_dict = {}
    val_index = i["val_index"]
    val_hinge_index = hinge_loss_fn(i["val_res"]["logit"] , i["val_res"]["labels"] )
    for j,k in zip(val_index,val_hinge_index):
        if j in val_dict:
            val_dict[j].append(k)
        else:
            val_dict[j] = [k]

    train_dict = {}
    train_index = i["train_index"]
    train_hinge_index = hinge_loss_fn(i["train_res"]["logit"] , i["train_res"]["labels"] )
    for j, k in zip(train_index,train_hinge_index):
        if j in train_dict:
            train_dict[j].append(k)
        else:
            train_dict[j] = [k]
    
    test_dict = {}
    test_index = i["test_index"]
    test_hinge_index = hinge_loss_fn(i["test_res"]["logit"] , i["test_res"]["labels"] )
    for j,k in zip(test_index,test_hinge_index):
        if j in test_dict:
            test_dict[j].append(k)
        else:
            test_dict[j] = [k]

    return (val_dict, train_dict, test_dict)

def plot_auc(name, target_val_score, target_train_score, epoch): 
    '''
    - Tính toán AUC (Area Under Curve) cho các mẫu đầu vào target_val_score và target_train_score.
    - Trả về diện tích dưới đường cong (AUC), diện tích dưới đường cong log (log_auc) và các giá trị TPR tại các ngưỡng FPR khác nhau.
    '''
    fpr, tpr, thresholds = metrics.roc_curve(torch.cat( [torch.zeros_like(target_val_score),torch.ones_like(target_train_score)] ).cpu().numpy(), torch.cat([target_val_score,target_train_score]).cpu().numpy())
    auc = metrics.auc(fpr, tpr)
    log_tpr, log_fpr = np.log10(tpr), np.log10(fpr)
    log_tpr[log_tpr < -5] =- 5
    log_fpr[log_fpr < -5] =- 5
    log_fpr = (log_fpr+5) / 5.0
    log_tpr = (log_tpr+5) / 5.0
    log_auc = metrics.auc(log_fpr,log_tpr)

    tprs = {}
    for fpr_thres in [10, 1, 0.1, 0.02, 0.01, 0.001, 0.0001]:
        tpr_index = np.sum(fpr < fpr_thres)
        tprs[str(fpr_thres)] = tpr[tpr_index-1]
    
    return auc, log_auc, tprs

# ==================== Vẽ đồ thị và lưu kết quả ====================

def fig_out(x_axis_data, MAX_CLIENTS, defence, seed, log_path, d, avg_d=None, single_score=None, other_scores=None, accs=None):
    '''
    Vẽ biểu đồ để thể hiện độ hiệu quả của các phương pháp tấn công khác nhau.
    '''
    
    # Định nghĩa màu cho các loại tấn công
    colors = {
        "cosine attack":"r", 
        "grad diff":"g",
        "loss based":"b",
        "grad norm":(242/256, 159/256, 5/256),
        "lira":"y",
        "log_lira":"k", 
        "lira_loss":'purple' 
    }
    
    # Định nghĩa nhãn cho các loại tấn công cơ bản (common)
    labels_per_epoch = {
        "cosine attack":"Grad-Cosine",
        "grad diff":"Grad-Diff",
        "loss based":"Blackbox-Loss",
        "grad norm":"Grad-Norm"
    }

    # Định nghĩa nhãn cho các phương pháp tấn công mới
    labels_temporal = {
        # Định nghĩa 2 nhãn để so sánh với 2 phương pháp tấn công của bài báo 
        "cosine attack":"Avg-Cosine", # Giá trị trung bình của cosine attack qua các epoch
        "loss based":"Loss-Series", # Giá trị trung bình của loss based qua các epoch

        # Định nghĩa 2 nhãn cho 2 phương pháp tấn công mới của bài báo
        "lira":"FedMIA-II",
        "lira_loss":"FedMIA-I"
    }

    fig = plt.figure(figsize = (6.5, 6.5), dpi = 200)
    fig.subplots_adjust(top = 0.91,
                        bottom = 0.160,
                        left = 0.180,
                        right = 0.9,
                        hspace = 0.2,
                        wspace = 0.2)
    # print_to_everything("\n==================== Tỉ lệ dự đoán đúng (TPR) tại tỉ lệ chịu lỗi 0.01 (@FPR = 0.01) ====================\n")
    for k in labels_per_epoch.keys():
        # print_to_everything("---> ", end='')
        # print_to_everything(k, d[k], end='\n\n')
        plt.plot(x_axis_data[0:len(d[k])], d[k], linewidth=1, label=labels_per_epoch[k], color=colors[k])
    plt.legend(loc = 3)  

    plt.xlim(-2, 305)
    my_x_ticks = np.arange(0, 302, 50)
    plt.xticks(my_x_ticks, size=14)
    if avg_d:
        for k in labels_temporal.keys():
            if avg_d[k]:    
                plt.hlines([avg_d[k]["0.001"]], xmin=0, xmax=300, label=labels_temporal[k], color=colors[k])

    plt.legend(prop={'size': 10})
    plt.xlabel('Epoch', fontsize = 14, fontdict = {'size': 14})  # x_label
    plt.ylabel('TPR@FPR=0.001', fontsize = 14, fontdict = {'size': 14})  # y_label
    plt.grid(axis='both')

    pdf_path=PATH.split("/")[0:-1]
    pdf_path="/".join(pdf_path)+f"/attack_fig_{select_mode}_{select_method}_n{SHADOW_NUM}_s{SEED}.pdf"
    print_to_console('\n===> Biểu đồ so sánh giữa các phương pháp tấn công lưu tại: ', pdf_path + '\n')
    plt.savefig(pdf_path)

    log_path = PATH.split("/")[0:-1]
    log_path = "/".join(log_path)+f"/attack_score_{select_mode}_{select_method}_n{SHADOW_NUM}_s{SEED}.log"
    
    # Lưu các chỉ số vào file JSON để dễ trích xuất
    with open(log_path,"w") as f:
        json.dump({"avg_d" : avg_d, "single_score" : single_score, "other_scores" : other_scores, "accs" : accs}, f, indent = 4)

# ==================== Các hàm chính cho việc tấn công ====================

def lira_attack_ldh_cosine(f, epch, K, save_dir, extract_fn = None, attack_mode = "cos"):
    '''
    Thực hiện tấn công FedMIA.
    Note: Có nhiều chỗ print bị em comment đi để ẩn đi các thông số không cần thiết để tránh rối mắt
    '''

    print_to_file(f"\n<----------- FedMIA Attack Result (mode {attack_mode}) ----------->\n")

    accs = []
    training_res = []
    for i in range(K):
        try:
            res = torch.load(f.format(i, epch))
            training_res.append(res)
            accs.append(res.get("test_acc", 0))
        except Exception as e:
            print_to_file(f"Failed to load client {i} at epoch {epch}: {e}")
            continue

    if len(training_res) < 2:
        print_to_file("Not enough training results for attack.")
        return accs, {}, 0, 0, ([], [])


    # print_to_file(f"a. Cấu trúc (kích thước) của vector loss sau khi tính toán trên dữ liệu test của các đối tượng\n")

    # Target model
    target_res = training_res[0]
    shadow_res = training_res[1:]
    if attack_mode == "cos":
        target_train_loss = np.array(target_res.get("tarin_cos", []))
        if MODE == "test":
            target_test_loss = np.array(target_res.get("test_cos", []))
        elif MODE == "val":
            target_test_loss = np.array(target_res.get("val_cos", []))
        elif MODE == "mix":
            test_cos = np.array(target_res.get("test_cos", []))
            mix_cos = np.array(target_res.get("mix_cos", []))
            if test_cos.size > 0:
                random_indices = np.random.permutation(test_cos.shape[0])[:mix_length]
                target_test_loss = np.concatenate([test_cos[random_indices], mix_cos], axis=0)
            else:
                target_test_loss = mix_cos
            # print_to_file('Client mục tiêu:', target_test_loss.shape)
    elif attack_mode == "diff":
        target_train_loss = np.array(target_res.get("tarin_diffs", []))
        if MODE == "test":
            target_test_loss = np.array(target_res.get("test_diffs", []))
        elif MODE == "val":
            target_test_loss = np.array(target_res.get("val_diffs", []))
        elif MODE == "mix":
            test_diffs = np.array(target_res.get("test_diffs", []))
            mix_diffs = np.array(target_res.get("mix_diffs", []))
            if test_diffs.size > 0:
                random_indices = np.random.permutation(test_diffs.shape[0])[:mix_length]
                target_test_loss = np.concatenate([test_diffs[random_indices], mix_diffs], axis=0)
            else:
                target_test_loss = mix_diffs
            # print_to_file('Client mục tiêu:', target_test_loss.shape)
    elif attack_mode == "loss":
        target_train_loss = -ce_loss_fn(
            target_res["train_res"]["logit"], target_res["train_res"]["labels"]
        ).cpu().numpy()
        if MODE == "test":
            target_test_loss = -ce_loss_fn(
                target_res["test_res"]["logit"], target_res["test_res"]["labels"]
            ).cpu().numpy()
        elif MODE == "val":
            target_test_loss = -ce_loss_fn(
                target_res["val_res"]["logit"], target_res["val_res"]["labels"]
            ).cpu().numpy()
        elif MODE == "mix":
            test_logit = target_res["test_res"]["logit"]
            test_labels = target_res["test_res"]["labels"]
            mix_logit = target_res["mix_res"]["logit"]
            mix_labels = target_res["mix_res"]["labels"]
            if test_logit.shape[0] > 0:
                random_indices = torch.randperm(test_logit.shape[0])[:mix_length]
                test_loss = -ce_loss_fn(test_logit[random_indices], test_labels[random_indices]).cpu().numpy()
                mix_loss = -ce_loss_fn(mix_logit, mix_labels).cpu().numpy()
                target_test_loss = np.concatenate([test_loss, mix_loss], axis=0)
            else:
                target_test_loss = -ce_loss_fn(mix_logit, mix_labels).cpu().numpy()
            # print_to_file('Client mục tiêu:', target_test_loss.shape)
    else:
        # print_to_file(f"Unknown attack_mode: {attack_mode}")
        return accs, {}, 0, 0, ([], [])

    # Shadow models
    shadow_train_losses = []
    shadow_test_losses = []
    for i, res in enumerate(shadow_res):
        if attack_mode == "cos":
            shadow_train_losses.append(np.array(res.get("tarin_cos", [])))
            if MODE == "val":
                shadow_test_losses.append(np.array(res.get("val_cos", [])))
            elif MODE == "test":
                shadow_test_losses.append(np.array(res.get("test_cos", [])))
            elif MODE == "mix":
                test_cos = np.array(res.get("test_cos", []))
                mix_cos = np.array(res.get("mix_cos", []))
                if test_cos.size > 0:
                    random_indices = np.random.permutation(test_cos.shape[0])[:mix_length]
                    shadow_test_loss = np.concatenate([test_cos[random_indices], mix_cos], axis=0)
                else:
                    shadow_test_loss = mix_cos
                # print_to_file(f'Client bóng thứ {i + 1}:', shadow_test_loss.shape)
                shadow_test_losses.append(shadow_test_loss)
        elif attack_mode == "diff":
            shadow_train_losses.append(np.array(res.get("tarin_diffs", [])))
            if MODE == "val":
                shadow_test_losses.append(np.array(res.get("val_diffs", [])))
            elif MODE == "test":
                shadow_test_losses.append(np.array(res.get("test_diffs", [])))
            elif MODE == "mix":
                test_diffs = np.array(res.get("test_diffs", []))
                mix_diffs = np.array(res.get("mix_diffs", []))
                if test_diffs.size > 0:
                    random_indices = np.random.permutation(test_diffs.shape[0])[:mix_length]
                    shadow_test_loss = np.concatenate([test_diffs[random_indices], mix_diffs], axis=0)
                else:
                    shadow_test_loss = mix_diffs
                # print_to_file(f'Client bóng thứ {i + 1}:', shadow_test_loss.shape)
                shadow_test_losses.append(shadow_test_loss)
        elif attack_mode == "loss":
            shadow_train_losses.append(
                -ce_loss_fn(res["train_res"]["logit"], res["train_res"]["labels"]).cpu().numpy()
            )
            if MODE == "val":
                shadow_test_losses.append(
                    -ce_loss_fn(res["val_res"]["logit"], res["val_res"]["labels"]).cpu().numpy()
                )
            elif MODE == "test":
                shadow_test_losses.append(
                    -ce_loss_fn(res["test_res"]["logit"], res["test_res"]["labels"]).cpu().numpy()
                )
            elif MODE == "mix":
                test_logit = res["test_res"]["logit"]
                test_labels = res["test_res"]["labels"]
                mix_logit = res["mix_res"]["logit"]
                mix_labels = res["mix_res"]["labels"]
                if test_logit.shape[0] > 0:
                    random_indices = torch.randperm(test_logit.shape[0])[:mix_length]
                    test_loss = -ce_loss_fn(test_logit[random_indices], test_labels[random_indices]).cpu().numpy()
                    mix_loss = -ce_loss_fn(mix_logit, mix_labels).cpu().numpy()
                    shadow_test_loss = np.concatenate([test_loss, mix_loss], axis=0)
                else:
                    shadow_test_loss = -ce_loss_fn(mix_logit, mix_labels).cpu().numpy()
                # print_to_file(f'Client bóng thứ {i + 1}:', shadow_test_loss.shape)
                shadow_test_losses.append(shadow_test_loss)
        else:
            # print_to_file(f"Unknown attack_mode: {attack_mode} for shadow model {i}")
            pass

    # Ensure all arrays are the same length for stacking
    min_train_len = min([arr.shape[0] for arr in shadow_train_losses] + [target_train_loss.shape[0]])
    min_test_len = min([arr.shape[0] for arr in shadow_test_losses] + [target_test_loss.shape[0]])
    target_train_loss = target_train_loss[:min_train_len]
    target_test_loss = target_test_loss[:min_test_len]
    shadow_train_losses = [arr[:min_train_len] for arr in shadow_train_losses]
    shadow_test_losses = [arr[:min_test_len] for arr in shadow_test_losses]
    shadow_train_losses_stack = np.vstack( shadow_train_losses )
    shadow_test_losses_stack = np.vstack( shadow_test_losses )

    # print_to_file('\nb. Giá trị trung bình (mean) và phương sai (var) của loss\n')
    # print_to_file(f"Target client train loss: mean = {target_train_loss.mean(axis=0)}, var = {target_train_loss.var(axis=0)}")
    # print_to_file(f"Target client test loss: mean = {target_test_loss.mean(axis=0)}, var = {target_test_loss.var(axis=0)}")

    # i = 1
    # for train_loss, test_loss in zip(shadow_train_losses, shadow_test_losses):
    #     print_to_file(f"Shadow client {i} train loss: mean = {train_loss.mean(axis=0)}, var = {train_loss.var(axis=0)}")
    #     print_to_file(f"Shadow client {i} test loss: mean = {test_loss.mean(axis=0)}, var = {test_loss.var(axis=0)}")
    #     i += 1

    # Define indices to view for easier inspection
    view_list = [0, 1, 2, 3, 4, 500, 501, 502, 503, 504, -5, -4, -3, -2, -1]

    def print_row(label, values, indices):
        print_to_file(f"{label:<10}", end='')
        for idx in indices:
            if idx < 0:
                idx = len(values) + idx
            if 0 <= idx < len(values):
                print_to_file(f"{values[idx]:>10.6f}", end='')
            else:
                print_to_file(f"{'N/A':>10}", end='')
        print_to_file('')

    # print_to_file('\nc. Training Samples Overview\n')
    # print_to_file(f"{'Index':<10}", end='')
    # for idx in view_list:
    #     print_to_file(f"{idx:>10}", end='')
    # print_to_file('')

    # print_row("Target", target_train_loss, view_list)
    # print_row("Mean", np.mean(shadow_train_losses_stack, axis=0), view_list)
    # print_row("Var", np.var(shadow_train_losses_stack, axis=0), view_list)
    # for i, train_loss in enumerate(shadow_train_losses, 1):
    #     print_row(f"Shadow{i}", train_loss, view_list)

    # print_to_file('\nd. Testing Samples Overview\n')
    # print_to_file(f"{'Index':<10}", end='')
    # for idx in view_list:
    #     print_to_file(f"{idx:>10}", end='')
    # print_to_file('')

    # print_row("Target", target_test_loss, view_list)
    # print_row("Mean", np.mean(shadow_test_losses_stack, axis=0), view_list)
    # print_row("Var", np.var(shadow_test_losses_stack, axis=0), view_list)
    # for i, test_loss in enumerate(shadow_test_losses, 1):
    #     print_row(f"Shadow{i}", test_loss, view_list)
    
    # print_to_file('\ne. Các cài đặt tấn công\n')
    # print_to_file('---> Cách tiền xử lý dữ liệu (none for iid / outlier for noniid):', select_method)
    # print_to_file('---> Tấn công dựa trên chỉ số (cos / diff / loss):', attack_mode)
    # print_to_file()

    # === Tính toán mean và var cho phân phối chuẩn hóa (normalization) ===
    # Nếu là chế độ non-iid (select_mode == 1) và attack_mode là 'cos', dùng phương pháp loại bỏ outlier
    if select_mode == 1 and attack_mode == 'cos' and select_method == 'outlier':
        # Khởi tạo mảng mean và var cho train/test
        train_mu_out = np.zeros(shadow_train_losses_stack.shape[1])
        train_var_out = np.zeros(shadow_train_losses_stack.shape[1])
        test_mu_out = np.zeros(shadow_test_losses_stack.shape[1])
        test_var_out = np.zeros(shadow_test_losses_stack.shape[1])

        # Loại bỏ outlier cho từng sample (theo cột)
        for j in range(shadow_train_losses_stack.shape[1]):
            col = shadow_train_losses_stack[:, j]
            mean = col.mean()
            std = col.std()
            # Giữ lại các giá trị không phải outlier (dưới mean + 3*std)
            filtered = col[col < mean + 3 * std]
            if filtered.size == 0:
                filtered = np.array([col.min()])
            train_mu_out[j] = filtered.mean()
            train_var_out[j] = filtered.var() + 1e-8

        for j in range(shadow_test_losses_stack.shape[1]):
            col = shadow_test_losses_stack[:, j]
            mean = col.mean()
            std = col.std()
            filtered = col[col < mean + 3 * std]
            if filtered.size == 0:
                filtered = np.array([col.min()])
            test_mu_out[j] = filtered.mean()
            test_var_out[j] = filtered.var() + 1e-8

    else:
        # Trường hợp còn lại: dùng mean/var thông thường trên toàn bộ shadow
        train_mu_out = shadow_train_losses_stack.mean(axis=0)
        train_var_out = shadow_train_losses_stack.var(axis=0) + 1e-8
        test_mu_out = shadow_test_losses_stack.mean(axis=0)
        test_var_out = shadow_test_losses_stack.var(axis=0) + 1e-8

    # In thông tin kiểm tra
    # print_to_file(f'f. Các thông số trong lần tấn công epoch {epch}\n')
    # print_to_file('---> Target train loss', target_train_loss[:10])
    # print_to_file('---> Train mu out:', train_mu_out[:10])
    # print_to_file('---> Target test loss:', target_test_loss[:10])
    # print_to_file('---> Test mu out:', test_mu_out[:10])

    # Tính xác suất (likelihood) cho mẫu train và test dựa trên phân phối chuẩn
    train_l_out = scipy.stats.norm.cdf(target_train_loss, train_mu_out, np.sqrt(train_var_out))
    test_l_out = scipy.stats.norm.cdf(target_test_loss, test_mu_out, np.sqrt(test_var_out))

    # print_to_file(f"---> Độ lệch chuẩn train: {np.sqrt(train_var_out).mean():.4f}, test: {np.sqrt(test_var_out).mean():.4f}")
    # print_to_file(f"---> Mean/Var train_l_out: {train_l_out.mean():.4f}/{train_l_out.var():.4f}")
    # print_to_file(f"---> Mean/Var test_l_out: {test_l_out.mean():.4f}/{test_l_out.var():.4f}")
    # print_to_file(f"---> Kích thước train_l_out: {train_l_out.shape}, test_l_out: {test_l_out.shape}")

    # Hiển thị một số giá trị mẫu để kiểm tra
    # print_to_file("---> Một số giá trị train_l_out:", train_l_out[:5])
    # print_to_file("---> Một số giá trị test_l_out:", test_l_out[:5])

    # Kiểm tra các chỉ số outlier trong test_l_out (giá trị lớn hơn 0.8)
    outlier_indices = np.where(test_l_out > 0.8)[0]
    # print_to_file(f"---> Số lượng outlier test_l_out > 0.8: {len(outlier_indices)}")
    # if len(outlier_indices) > 0:
    #     print_to_file("---> Một số giá trị outlier test_l_out:", test_l_out[outlier_indices[:5]])
    #     print_to_file("---> Tương ứng target_test_loss:", target_test_loss[outlier_indices[:5]])
    #     print_to_file("---> Tương ứng shadow_test_losses_stack:\n", shadow_test_losses_stack[:, outlier_indices[:5]])

    # Kiểm tra các chỉ số outlier trong train_l_out (giá trị nhỏ hơn 0.5)
    mem_outlier_indices = np.where(train_l_out < 0.5)[0]
    # print_to_file(f"---> Số lượng outlier train_l_out < 0.5: {len(mem_outlier_indices)}")
    # if len(mem_outlier_indices) > 0:
    #     print_to_file("---> Một số giá trị outlier train_l_out:", train_l_out[mem_outlier_indices[:5]])
    #     print_to_file("---> Tương ứng target_train_loss:", target_train_loss[mem_outlier_indices[:5]])
    #     print_to_file("---> Tương ứng shadow_train_losses_stack:\n", shadow_train_losses_stack[:, mem_outlier_indices[:5]])

    auc, log_auc, tprs = plot_auc("lira", torch.tensor(test_l_out), torch.tensor(train_l_out), epch)
    
    print_to_file(f"---> True positive rate at different false positive rate: {tprs}\n---> Log AUC: {log_auc:.4f}")

    return accs, tprs, auc, log_auc, (train_l_out, test_l_out)

def cos_attack(f, K, epch, attack_mode, extract_fn = None):

    accs = []
    target_res = torch.load(f.format(0,epch))
    tprs = None

    print_to_file(f"\n<----------- Common Attack Result ({attack_mode}) ----------->\n")

    if attack_mode == "cosine attack":
        if MODE=="test":
            val_liratios=target_res['test_cos']
        elif MODE=="val":
            val_liratios=target_res['val_cos']
        elif MODE=='mix':
            random_indices = torch.randperm(target_res["test_cos"].shape[0])
            val_liratios = target_res["test_cos"][random_indices[:mix_length]]
            val_liratios = torch.tensor(val_liratios)
            mix_test_loss = torch.tensor(target_res["mix_cos"])
            mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
            val_liratios = mix_test_loss

        val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=target_res['tarin_cos']
        train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("cos_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)
  
        print_to_file(f"---> True positive rate at different false positive rate: {tprs}\n---> Log AUC: {log_auc:.4f}")

    elif attack_mode == "grad diff":
        if MODE=="test":
            val_liratios=target_res['test_diffs']
        elif MODE=="val":
            val_liratios=target_res['val_diffs']
        elif MODE=='mix':
            random_indices = torch.randperm(target_res["test_diffs"].shape[0])
            val_liratios = target_res["test_diffs"][random_indices[:mix_length]]
            val_liratios = torch.tensor(val_liratios)
            mix_test_loss = torch.tensor(target_res["mix_diffs"])
            mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
            val_liratios = mix_test_loss
        val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=target_res['tarin_diffs']
        train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("diff_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)   
        
        print_to_file(f"---> True positive rate at different false positive rate: {tprs}\n---> Log AUC: {log_auc:.4f}")

    elif attack_mode == "grad norm":
        if MODE=="test":
            val_liratios=target_res['test_grad_norm']
        elif MODE=="val":
            val_liratios=target_res['val_grad_norm']
        elif MODE=='mix':
            random_indices = torch.randperm(target_res["test_grad_norm"].shape[0])
            val_liratios = target_res["test_grad_norm"][random_indices[:mix_length]]
            val_liratios = -torch.tensor(val_liratios)
            mix_test_loss = -torch.tensor(target_res["mix_grad_norm"])
            mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
            # print_to_file('mix_test_loss shape:',mix_test_loss.shape)
            val_liratios = mix_test_loss
        val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=target_res['tarin_grad_norm']
        train_liratios=-np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("grad_norm_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch) 

        print_to_file(f"---> True positive rate at different false positive rate: {tprs}\n---> Log AUC: {log_auc:.4f}")

    elif attack_mode == "loss based":
        if MODE=="test":
            val_liratios=-ce_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] )
        elif MODE=="val":
            val_liratios=-ce_loss_fn(target_res["val_res"]["logit"] , target_res["val_res"]["labels"] )
        elif MODE =='mix':
            random_indices = torch.randperm(target_res["test_res"]["logit"].shape[0])
            val_liratios =-ce_loss_fn(target_res["test_res"]["logit"][random_indices[:mix_length]],\
                                            target_res["test_res"]["labels"][random_indices[:mix_length]])
            mix_test_loss=-ce_loss_fn(target_res["mix_res"]["logit"] , target_res["mix_res"]["labels"] ).cpu().numpy()
            mix_test_loss = np.concatenate([val_liratios,mix_test_loss],axis=0)
            # print_to_file('mix_test_loss shape:',mix_test_loss.shape)
            val_liratios = mix_test_loss

        # val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=-ce_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] )
        # train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("loss_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)    

        print_to_file(f"---> True positive rate at different false positive rate: {tprs}\n---> Log AUC: {log_auc:.4f}")
    
    return accs, tprs, auc, log_auc, (train_liratios, val_liratios)

# Hàm thực thi và so sánh các dạng tấn công
@ torch.no_grad()
def attack_comparison(p, log_path, save_dir, epochs, MAX_CLIENTS, defence, seed):
    """
    Summary of the Correspondence between attack methods in paper and scores in codea:
    summary_dict={

    'Blackbox-Loss': scores['loss based'],
    'Grad-Cosine': scores['cosine attack'],
    'Grad-Diff': scores['grad diff'],
    'Grad-Norm': scores['grad norm'],

    'Loss-Series': avg_scores["loss based"], 
    'Avg-Cosine': avg_scores["cosine attack"],

    'FedMIA-I': avg_scores["lira_loss"],
    'FedMIA-II': avg_scores["lira"]
    }
    """

    lira_loss_scores = [] 
    lira_scores = [] 
    common_scores = [] 
    other_scores = {} 

    scores = { k : [] for k in baseline_attack_modes}
    scores["lira"] = []
    scores["lira_loss"] = []
    single_score = { k : 0 for k in baseline_attack_modes}
    single_score["lira"] = 0
    single_score["lira_loss"] = 0
    reses_lira = []
    reses_lira_loss = []
    reses_common = {k : [] for k in baseline_attack_modes}
    avg_scores = {k : None for k in baseline_attack_modes}
    avg_scores["lira"] = None
    avg_scores["lira_loss"] = None

    auc_dict = {k : [] for k in baseline_attack_modes}
    auc_dict["lira"] = []
    auc_dict["lira_loss"] = []

    final_acc = None

    for epch in epochs:
        print_to_file(f'\n==================== Attack at epoch {epch} ====================')
        lira_score = lira_attack_ldh_cosine(p,epch,MAX_CLIENTS,save_dir, extract_fn=extract_hinge_loss, attack_mode = 'cos') # Dựa trên cosine
        lira_loss_score = lira_attack_ldh_cosine(p,epch,MAX_CLIENTS,save_dir, extract_fn=extract_hinge_loss, attack_mode ='loss') # Dựa trên loss
        scores["lira"].append(lira_score[1]['0.001'])
        scores["lira_loss"].append(lira_loss_score[1]['0.001'])
        auc_dict["lira"].append(lira_score[2])
        auc_dict["lira_loss"].append(lira_loss_score[2])
        for attack_mode in baseline_attack_modes:
            common_score = cos_attack(p, 0, epch, attack_mode, extract_fn = extract_hinge_loss) 
            reses_common[attack_mode].append(common_score[-1])
            scores[attack_mode].append(common_score[1]['0.001'])
            auc_dict[attack_mode].append(common_score[2])
            if epch == 200 and attack_mode == "loss based":
                other_scores["loss_single_epch_score"]=common_score[1] # tpr
                other_scores["loss_single_auc"]=[common_score[2],common_score[3]] # tpr, auc

        lira_scores.append(lira_score[1]['0.001'])
        lira_loss_scores.append(lira_loss_score[1]['0.001'])
        common_scores.append(common_score[1]['0.001'])
        reses_lira.append(lira_score[-1])  
        reses_lira_loss.append(lira_loss_score[-1])
        final_acc = lira_score[0]

    for attack_mode in baseline_attack_modes:
        sorted_id = sorted(range(len(scores[attack_mode])), key=lambda k: scores[attack_mode][k], reverse=True)
        single_score[attack_mode]=(scores[attack_mode][sorted_id[0]])
        single_score[f'single {attack_mode}_auc'] = auc_dict[attack_mode][sorted_id[0]]

    for attack_mode in ['lira', 'lira_loss']:
        sorted_id = sorted(range(len(scores[attack_mode])), key=lambda k: scores[attack_mode][k], reverse=True)
        single_score[attack_mode]=(scores[attack_mode][sorted_id[0]])
        single_score[f'single {attack_mode}_auc'] = auc_dict[attack_mode][sorted_id[0]]

    for attack_mode in baseline_attack_modes:
        # print_to_file('len(scores[attack_mode]): ',len(scores[attack_mode]))  30
        single_score[f'200 {attack_mode}']=(scores[attack_mode][int(epochs[-1]/20)])
        single_score[f'200 single_{attack_mode}_auc'] = auc_dict[attack_mode][int(epochs[-1]/20)]
    
    for attack_mode in ['lira', 'lira_loss']:
        single_score[f'200 {attack_mode}']=(scores[attack_mode][int(epochs[-1]/20)])
        single_score[f'200 single_{attack_mode}_auc'] = auc_dict[attack_mode][int(epochs[-1]/20)]

    print_to_file('\n==================== Tổng hợp kết quả các phương pháp tấn công ====================\n')

    # 1. FedMIA-I (LIRA - Loss)
    reses = reses_lira_loss
    train_score = np.vstack([i[0].reshape(1, -1) for i in reses]).mean(axis=0)
    test_score = np.vstack([i[1].reshape(1, -1) for i in reses]).mean(axis=0)
    auc, log_auc, tprs = plot_auc("FedMIA-I (LIRA - Loss)", torch.tensor(test_score), torch.tensor(train_score), 999)
    print_to_everything("---------- FedMIA-I (LIRA - Loss) ----------")
    print_to_everything(f"  - Trung bình train score: {np.mean(train_score):.4f} (var: {np.var(train_score):.4f})")
    print_to_everything(f"  - Trung bình test score: {np.mean(test_score):.4f} (var: {np.var(test_score):.4f})")
    print_to_everything(f"  - TPR tại các FPR: {tprs}")
    print_to_everything(f"  - AUC: {auc:.4f}, Log AUC: {log_auc:.4f}\n")
    avg_scores["lira_loss"] = tprs
    other_scores["lira_loss_auc"] = [auc, log_auc]

    # 2. FedMIA-II (LIRA - Cosine)
    reses = reses_lira
    train_score = np.vstack([i[0].reshape(1, -1) for i in reses]).mean(axis=0)
    test_score = np.vstack([i[1].reshape(1, -1) for i in reses]).mean(axis=0)
    auc, log_auc, tprs = plot_auc("FedMIA-II (LIRA - Cosine)", torch.tensor(test_score), torch.tensor(train_score), 999)
    print_to_everything("---------- FedMIA-II (LIRA - Cosine) ----------")
    print_to_everything(f"  - Trung bình train score: {np.mean(train_score):.4f} (var: {np.var(train_score):.4f})")
    print_to_everything(f"  - Trung bình test score: {np.mean(test_score):.4f} (var: {np.var(test_score):.4f})")
    print_to_everything(f"  - TPR tại các FPR: {tprs}")
    print_to_everything(f"  - AUC: {auc:.4f}, Log AUC: {log_auc:.4f}\n")
    avg_scores["lira"] = tprs
    other_scores["lira_auc"] = [auc, log_auc]

    # 3. Grad-Cosine (Avg-Cosine)
    reses = reses_common["cosine attack"]
    train_score = np.vstack([i[0].reshape(1, -1) for i in reses]).mean(axis=0)
    test_score = np.vstack([i[1].reshape(1, -1) for i in reses]).mean(axis=0)
    auc, log_auc, tprs = plot_auc("Grad-Cosine", torch.tensor(test_score), torch.tensor(train_score), 999)
    print_to_everything("---------- Grad-Cosine (Trung bình) ----------")
    print_to_everything(f"  - Trung bình train score: {np.mean(train_score):.4f} (var: {np.var(train_score):.4f})")
    print_to_everything(f"  - Trung bình test score: {np.mean(test_score):.4f} (var: {np.var(test_score):.4f})")
    print_to_everything(f"  - TPR tại các FPR: {tprs}")
    print_to_everything(f"  - AUC: {auc:.4f}, Log AUC: {log_auc:.4f}\n")
    avg_scores["cosine attack"] = tprs
    other_scores["cos_attack_auc"] = [auc, log_auc]

    # 4. Grad-Diff (Trung bình)
    reses = reses_common["grad diff"]
    train_score = np.vstack([i[0].reshape(1, -1) for i in reses]).mean(axis=0)
    test_score = np.vstack([i[1].reshape(1, -1) for i in reses]).mean(axis=0)
    auc, log_auc, tprs = plot_auc("Grad-Diff", torch.tensor(test_score), torch.tensor(train_score), 999)
    print_to_everything("---------- Grad-Diff (Trung bình) ----------")
    print_to_everything(f"  - Trung bình train score: {np.mean(train_score):.4f} (var: {np.var(train_score):.4f})")
    print_to_everything(f"  - Trung bình test score: {np.mean(test_score):.4f} (var: {np.var(test_score):.4f})")
    print_to_everything(f"  - TPR tại các FPR: {tprs}")
    print_to_everything(f"  - AUC: {auc:.4f}, Log AUC: {log_auc:.4f}\n")
    avg_scores["grad diff"] = tprs
    other_scores["grad_diff_auc"] = [auc, log_auc]

    # 5. Grad-Norm (Trung bình)
    reses = reses_common["grad norm"]
    train_score = -np.vstack([i[0].reshape(1, -1) for i in reses]).mean(axis=0)
    test_score = -np.vstack([i[1].reshape(1, -1) for i in reses]).mean(axis=0)
    auc, log_auc, tprs = plot_auc("Grad-Norm", torch.tensor(test_score), torch.tensor(train_score), 999)
    print_to_everything("---------- Grad-Norm (Trung bình) ----------")
    print_to_everything(f"  - Trung bình train score: {np.mean(train_score):.4f} (var: {np.var(train_score):.4f})")
    print_to_everything(f"  - Trung bình test score: {np.mean(test_score):.4f} (var: {np.var(test_score):.4f})")
    print_to_everything(f"  - TPR tại các FPR: {tprs}")
    print_to_everything(f"  - AUC: {auc:.4f}, Log AUC: {log_auc:.4f}\n")
    avg_scores["grad norm"] = tprs
    other_scores["grad_norm_auc"] = [auc, log_auc]

    # 6. Blackbox-Loss (Trung bình)
    reses = reses_common["loss based"]
    train_score = np.vstack([i[0].reshape(1, -1) for i in reses]).mean(axis=0)
    test_score = np.vstack([i[1].reshape(1, -1) for i in reses]).mean(axis=0)
    auc, log_auc, tprs = plot_auc("Blackbox-Loss", torch.tensor(test_score), torch.tensor(train_score), 999)
    print_to_everything("---------- Blackbox-Loss (Trung bình) ----------")
    print_to_everything(f"  - Trung bình train score: {np.mean(train_score):.4f} (var: {np.var(train_score):.4f})")
    print_to_everything(f"  - Trung bình test score: {np.mean(test_score):.4f} (var: {np.var(test_score):.4f})")
    print_to_everything(f"  - TPR tại các FPR: {tprs}")
    print_to_everything(f"  - AUC: {auc:.4f}, Log AUC: {log_auc:.4f}\n")
    avg_scores["loss based"] = tprs
    other_scores["loss_based_auc"] = [auc, log_auc]

    fig_out(epochs, MAX_CLIENTS, defence, seed, log_path, scores, avg_scores, single_score, other_scores, final_acc)

# ==================== Hàm chính ====================

def main(argv):

    # ---------- Khai báo và khởi tạo các biến ----------

    global MODE, baseline_attack_modes, PATH, p_folder, device, select_mode, select_method, SHADOW_NUM, SEED, mix_length, SAVE_DIR # Khai báo biến toàn cục cho tiện
    baseline_attack_modes = ["cosine attack", "grad diff", "loss based", "grad norm"] # Các phương pháp tấn công cơ bản, dùng để so sánh với 2 phương pháp tấn công mới của bài báo
    p_folder = argv[1]  # Thư mục chứa log training và attacking 
    PATH = argv[1] # Biến dùng để thao tác với thư mục
    epochs = list(range(10, int(argv[2]) + 1, 10)) # Chỉ lấy các epoch là bội số của 10 để phục vụ tấn công nhằm tiết kiệm tài nguyên
    MODE = argv[3] # Chen chế tấn công (train/test/val/mix)
    device = argv[4] # Index của thiết bị sử dụng (cpu/gpu)
    SEED = int(argv[5]) # Đặt seed để đảm bảo ngẫu nhiên của các thuật toán random
    MAX_CLIENTS = 10 # Số lượng client tối đa 
    
    for root, dirs, files in os.walk(p_folder, topdown=False):
        for name in dirs:

            # Nếu không phải là thư mục chứa log training thì bỏ qua
            if  root != p_folder: 
                continue

            # Xóa final model khỏi scoop tấn công
            if '_final_model' in name:
                continue
            
            PATH = os.path.join(root, name) # Đường dẫn đến thư mục chứa log training
            PATH += "/client_{}_losses_epoch{}.pkl" # Đường dẫn đến file log của client

            # Trích xuất thông tin từ tên thư mục
            MAX_CLIENTS = int(name.split("_K")[1].split("_")[0])
            model = name.split("_")[3]
            defence = name.split("_")[-5].strip('def').strip('0.0')
            seed = name.split("_")[-1]
            save_dir = p_folder + '/' + name
            SAVE_DIR = save_dir

            if 'iid$1' in name:
                select_mode = 0
                select_method = 'none'
                SHADOW_NUM = 9
            else:
                select_mode = 1
                select_method = 'outlier'
                SHADOW_NUM = 4
            

            print_to_everything(f"1. Thư mục log: {os.path.join(root, name)}")
            print_to_everything(f"2. Chế độ tấn công: {MODE}")
            print_to_everything(f"3. Định dạng các file model: {PATH}")
            print_to_everything(f"4. Thư mục gốc: {p_folder}")
            print_to_everything(f"5. Cách xử lý loss: {select_method}")
            print_to_everything(f"8. Số lượng client trong mô hình bóng: {SHADOW_NUM}")
            print_to_everything(f"9. Seed: {SEED}")
            print_to_everything(f"10. Model được sử dụng: {model}\n")


            if 'cifar100' in name:
                mix_length = int(10000/MAX_CLIENTS)
            elif 'cicmaldroid' in name:
                mix_length = 116 #232

            if model == "alexnet":
                log_path="logs/log_alex"
            elif model == "mlp":
                log_path="logs/log_mlp"
            elif model == "resnet":
                log_path="logs/log_res"

            print_to_everything("---> Attacking...")

            try:
                print_to_console("\n#################### Kết quả tấn công ####################\n")
                attack_comparison(PATH, log_path, save_dir, epochs, MAX_CLIENTS, defence, seed)
                print_to_console("===> Tấn công thành công! Kiểm tra file log để xem chi tiết!\n")
            except IOError:
                # Loại bỏ thông báo lỗi do kiểm tra file có tồn tại hay không trước khi truyền tham số vào tên file
                if PATH != "log_fedmia/iid/cicmaldroid_K10_N5000_mlp_defnone_iid$1_$1_$sgd_local1_s18052025mlp/client_{}_losses_epoch{}.pkl":
                    print_to_console(f"===> OMG! File {PATH} not found!\n")

if __name__ == "__main__":
    main(sys.argv)