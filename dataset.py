'''
Adapted from Pytorch official code
CIFAR dataset classes that allow customized partition
'''

from PIL import Image
import os
import os.path
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
import pandas as pd
import io

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive



class CIFAR10(VisionDataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    full_list = train_list + test_list
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, indices, transform=None, target_transform=None, download=False,need_index=False):

        super(CIFAR10, self).__init__(root,
                                      transform=transform,
                                      target_transform=target_transform)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.data = []
        self.targets = []
        self.indices = indices #
        self.need_index=need_index #
        

        # now load the picked numpy arrays
        for file_name, checksum in self.full_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data = entry['data']
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.data = self.data[self.indices]  
        self.targets = np.array(self.targets)[self.indices] 
        self.true_index = np.array([i for i in range(60000)])[self.indices] # The serial number of self.data in the original data set and the index of the data itself are also required.

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        i=self.true_index[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.need_index:
            return img, target, i
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

        This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    full_list = train_list + test_list
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class CICMalDroidDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None, mode="multiclass"):
        """
        Args:
            csv_file (string): Đường dẫn tới file CSV chứa feature vectors.
            transform (callable, optional): Hàm transform sẽ được áp dụng lên features.
            target_transform (callable, optional): Hàm transform sẽ được áp dụng lên nhãn.
            mode (str): 'multiclass' để sử dụng 5 lớp, hoặc 'binary' để phân loại độc hại/lành tính.
        """
        # Đọc file và loại bỏ các dòng bắt đầu bằng "//"
        with open(csv_file, "r") as f:
            lines = f.readlines()
        filtered_lines = [line for line in lines if not line.lstrip().startswith("//")]
        self.dataframe = pd.read_csv(io.StringIO("".join(filtered_lines)))
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        # Giả sử cột "Class" chứa nhãn (có thể là dạng chuỗi) và các cột khác là đặc trưng
        if "Class" not in self.dataframe.columns:
            raise ValueError("Không tìm thấy cột 'Class' trong file CSV!")

        # Nhãn gốc dùng cho bài toán phân loại 5 lớp
        self.labels = self.dataframe["Class"]

        # Nếu bạn muốn thực hiện bài toán phân loại nhị phân:
        # Giả sử rằng chỉ nhãn 'Benign' được xem là lành tính, mọi nhãn khác là độc hại.
        if mode == "binary":
            self.labels = self.labels.apply(lambda x: 0 if x.strip().lower() == "benign" else 1)
        else:
            # Tiến hành mapping các lớp thành chỉ số số nguyên (ví dụ, sắp xếp theo thứ tự xuất hiện)
            unique_labels = self.labels.unique()
            self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self.labels = self.labels.map(self.label_map)

        # Đặc trưng là tất cả các cột trừ cột "Class"
        self.features = self.dataframe.drop(columns=["Class"]).values.astype(float)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # Lấy vector đặc trưng và nhãn tại index chỉ định
        feature = self.features[index]
        label = self.labels.iloc[index] if isinstance(self.labels, pd.Series) else self.labels[index]

        # Chuyển đổi vector đặc trưng thành tensor float
        sample = torch.tensor(feature, dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        return sample, label