import wandb
from torch.utils.data import Dataset
import tqdm
import pickle
import torch
import os
from skimage.transform import resize
import torchvision.transforms as transforms
import torch.nn.functional as F
import random

class ADNIDataset(Dataset):
    def __init__(self, dataset, data_dir, resize_dim=None, pin_memory=False, train=False):
        self.dataset = dataset
        self.data_dir = data_dir
        if resize_dim is None:
            resize_dim = 256
        self.resize_dim = resize_dim
        self.pin_memory = pin_memory
        self.images = []

        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=self.resize_dim, scale=(0.9, 1.1), ratio=(1.0, 1.0),antialias=True),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize_dim,antialias=True),
            ])

    def __getitem__(self, idx):
        if not self.pin_memory:
            with open(os.path.join(self.data_dir, self.dataset[idx]), 'rb') as f:
                sample = pickle.load(f)
                X1, y1, X2, y2 = sample['X1'], sample['y1'], sample['X2'], sample['y2']
        else:
            image = self.images[idx]

        X1 = self.tensor_transforms(X1)/256
        X2 = self.tensor_transforms(X2)/256

        return X1.float(), torch.tensor([y1], dtype=torch.long), X2.float(), torch.tensor([y2], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


    # def _load_images(self):
    #     filenames = self.dataset['image']
    #     for filename in tqdm.tqdm(filenames, desc='Loading dataset to memory'):
    #         image = nib.load(filename)
    #         image = nib.as_closest_canonical(image).get_fdata()
    #         self.images.append(image)
import numpy as np
import ruptures as rpt
class MRIDataset(Dataset):
    def __init__(self, data_dir, resize_dim, train=False):
        self.data_dir = data_dir
        self.age=[]#this is clinical age
        if resize_dim is None:
            resize_dim = 256
        self.resize_dim = resize_dim
        self.precision = 1
        self.data_info = self._load_data_info()
        self.K=0
        self.cluster_means = np.zeros((999,resize_dim))

        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=self.resize_dim, scale=(0.9, 1.1), ratio=(1.0, 1.0),antialias=True),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize_dim,antialias=True),
            ])



    def _load_data_info(self):
        # Load all combinations of rid and j, and store path information
        data_info = []
        self.precision=1#i.e., how many clusters per year
        self.age = []
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith('sample_'):
                parts = dir_name.split('_')
                rid, j, age = parts[1], parts[2], parts[3]
                dir_path = os.path.join(self.data_dir, dir_name)
                data_info.append({
                    'dir_path': dir_path,
                    'rid': rid,
                    'j': int(j),
                    'age': float(age)
                })
                self.age.append(int(float(age)))
        self.K=0
        return data_info
    def update_label_first(self, rid, j, age,age_hat1_, feature):
        """
        Input rid, j, age are vectors of length n, feature is an n*d matrix.
        Sort feature, j, rid according to age size.
        Then perform change point detection on feature using l2norm and pelt.
        Get the number of change points K and their positions.
        """
        # Step 1: Sort rid, j, age, feature by age
        sorted_indices = np.argsort(age_hat1_)
        rid_sorted = np.array(rid)[sorted_indices]
        j_sorted = np.array(j)[sorted_indices]
        age_sorted=np.array(age)[sorted_indices]
        feature_sorted = feature[sorted_indices]

        # Step 2: Perform change point detection on sorted features
        model = rpt.Pelt(model="l2", min_size=3).fit(feature)

        # Use predict method for change point detection, return breakpoint list

        # Initialize pen, gradually adjust pen to find approximately 40 intervals
        pen = 1
        detected_breakpoints = []
        while len(detected_breakpoints) < 30- 1: # number of intervals = number of breakpoints + 1
            detected_breakpoints = model.predict(pen=pen)
            pen /= 1.1  # if too few breakpoints, decrease pen
        while len(detected_breakpoints) > 40: # number of intervals = number of breakpoints + 1
            detected_breakpoints = model.predict(pen=pen)
            pen *= 1.1  # if too many breakpoints, increase pen

        # Step 3: Calculate the number and positions of change points
        K = len(detected_breakpoints)   # 变点个数为区间数 - 1
        # breakpoints = detected_breakpoints[:-1]  # 最后一个是结尾位置，去掉它
        segment_means = np.zeros((K, feature_sorted.shape[1]))  # K*d的nparray
        start = 0
        cluster_labels = np.zeros(len(age_sorted), dtype=int)

        for i, bp in enumerate(detected_breakpoints):
            segment_means[i, :] = feature_sorted[start:bp].mean(axis=0)

            cluster_labels[start:bp] = i  # 将当前区间内的样本标记为所属的簇
            start = bp
        # 返回排序后的rid, j, clinical_age, feature，以及变点个数和位置,更新dataset
        self.K=K
        self.cluster_means = segment_means
        data_info = []
        for i in range(len(j_sorted )):
            dir_name='sample_'+str(rid_sorted[i])+'_'+str(j_sorted[i])+'_'+str(age_sorted[i])
            dir_path = os.path.join(self.data_dir, dir_name)
            data_info.append({
                'dir_path': dir_path,
                'rid': rid_sorted[i],
                'j': int(j_sorted[i]),
                'age': float(age_sorted[i])
            })
        self.age=cluster_labels
        self.data_info=data_info

        return  K,  detected_breakpoints

    def update_label(self, rid, j, age, feature):
        """此时已是第一个epoch训练完毕，即我们已知每个cluster的特征是多少。"""
        # 初始化存储结果的列表
        cluster_assignments = []
        rid_new=[]
        j_new=[]
        age_new=[]
        feature_new=[]
        for rid_value in set(rid):  # 针对每个rid分别求解一个优化模型
            # 取出相应rid的子集索引
            subset_index_i = [index for index, value in enumerate(rid) if value == rid_value]
            feature_sub = feature[subset_index_i]
            j_sub = np.array(j)[subset_index_i]
            age_sub = np.array(age)[subset_index_i]

            # 按照age对子集中的j, age, feature进行排序
            sorted_indices = np.argsort(age_sub)
            j_sorted = j_sub[sorted_indices]
            age_sorted = age_sub[sorted_indices]
            feature_sorted = feature_sub[sorted_indices]

            best_cluster = None
            best_loss = float('inf')

            for k in range(self.K):
                loss = 0  # 计算其第一张图assign到k时的初始loss
                pre_age = age_sorted[0]

                for idx in range(len(age_sorted)):
                    gap =  age_sorted[idx]-pre_age
                    adjusted_cluster = int(k + gap * self.precision)
                    if adjusted_cluster>self.K-1:
                        loss = 999999
                        break
                    loss += F.mse_loss(torch.tensor(self.cluster_means[adjusted_cluster]),
                                       torch.tensor(feature_sorted[idx]))
                    # pre_age = age_sorted[idx]

                # 如果当前簇的loss小于之前计算的最优loss，则更新最优簇分配
                if loss < best_loss:
                    best_loss = loss
                    best_cluster = k

            # 存储该rid子集的最优簇分配

            for idx in range(len(age_sorted)):#把新分配的cluster label 存储起来
                rid_new.append(rid_value)
                j_new.append(j_sorted[idx])
                age_new.append(age_sorted[idx])
                gap = age_sorted[idx] - pre_age
                cluster_assignments.append( int(best_cluster+ gap * self.precision))
                feature_new.append(feature_sorted[idx])
        feature_new=np.vstack(feature_new)
        segment_means = np.zeros((self.K, feature_new.shape[1]))  # K*d的nparray
        for i in range(self.K):
            if sum(np.array(cluster_assignments)==i)>0:
                self.cluster_means[i, :] = feature_new[np.array(cluster_assignments)==i].mean(axis=0)
        self.segment_means = segment_means
        self.rid=rid_new
        self.j=j_new
        self.age=cluster_assignments
        data_info=[]
        for i in range(len(j_new )):
            dir_name='sample_'+str(rid_new[i])+'_'+str(j_new[i])+'_'+str(age_new[i])
            dir_path = os.path.join(self.data_dir, dir_name)
            data_info.append({
                'dir_path': dir_path,
                'rid': rid_new[i],
                'j': int(j_new[i]),
                'age': float(age_new[i])
            })
        return cluster_assignments


    def __len__(self):
        # 返回不同 rid 和 j 的组合的数量
        return len(self.data_info)

    def __getitem__(self, index,stage=0):#img2是相同rid但不同j，img3是不同的rid
        # 获取当前index对应的rid和j组合
        selected_data = self.data_info[index]

        # 获取该rid和j组合下的所有图像
        images = os.listdir(selected_data['dir_path'])
        selected_image = random.choice(images)
        with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
            img1 = pickle.load(f)['X']
        age1 = selected_data['age']
        clinical_age1=self.age[index]
        # 找到相同rid但不同j的所有索引
        same_rid_different_j_indices = [
            idx for idx, info in enumerate(self.data_info)
            if info['rid'] == selected_data['rid'] and info['j'] != selected_data['j']
        ]

        # 如果存在不同j的情况，随机选择一个索引
        if same_rid_different_j_indices:
            different_j_index = random.choice(same_rid_different_j_indices)
            selected_image_info_j = self.data_info[different_j_index]
            images_j = os.listdir(selected_image_info_j['dir_path'])
            selected_image_j = random.choice(images_j)
            with open(os.path.join(selected_image_info_j['dir_path'], selected_image_j), 'rb') as f:
                img2 = pickle.load(f)['X']
            age2 = selected_image_info_j['age']
        else:
            img2 = img1  # 如果没有不同j的情况，可以选择返回相同的图像
            age2 = age1
            different_j_index = index
        clinical_age2 = self.age[different_j_index]
        # 随机选择一个不同rid的索引
        different_rid_index = index
        while True:
            different_rid_index = random.randint(0, len(self.data_info) - 1)
            if self.data_info[different_rid_index]['rid'] != selected_data['rid']:
                break

        # 使用不同rid的索引获取图像
        same_age_different_rid_indices = [
            i for i, info in enumerate(self.data_info)
            if self.age[i] == self.age[index] and info['rid'] != selected_data['rid']
        ]
        if same_age_different_rid_indices:
        # 随机选择一个符合条件的索引
            different_rid_index = random.choice(same_age_different_rid_indices)
        else:
            same_age_different_rid_indices = [
                i for i, info in enumerate(self.data_info)
                if abs(self.age[i] - self.age[index])<2 and info['rid'] != selected_data['rid']
            ]

        # 使用不同rid的索引获取图像
        selected_image_info_rid = self.data_info[different_rid_index]
        images_rid = os.listdir(selected_image_info_rid['dir_path'])
        selected_image_rid = random.choice(images_rid)
        with open(os.path.join(selected_image_info_rid['dir_path'], selected_image_rid), 'rb') as f:
            img3 = pickle.load(f)['X']
        age3 = selected_image_info_rid['age']
        clinical_age3 = self.age[different_rid_index]

        img2 = self.tensor_transforms(img2)/256
        img3 = self.tensor_transforms(img3)/256
        img1 = self.tensor_transforms(img1) / 256

        return img1.float(), age1, index, img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index,clinical_age1,clinical_age2,clinical_age3,self.cluster_means[clinical_age1],self.cluster_means[clinical_age2],self.cluster_means[clinical_age3]
class MRIDataset_three(Dataset):
    def __init__(self, data_dir, resize_dim, train=False):
        self.data_dir = data_dir
        self.age=[]#this is clinical age
        if resize_dim is None:
            resize_dim = 256
        self.resize_dim = resize_dim
        self.precision = 1
        self.data_info = self._load_data_info()
        self.K=0
        self.cluster_means = np.zeros((999,resize_dim))

        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=self.resize_dim, scale=(0.9, 1.1), ratio=(1.0, 1.0),antialias=True),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize_dim,antialias=True),
            ])



    def _load_data_info(self):
        # Load all combinations of rid and j, and store path information
        data_info = []
        self.precision=1#i.e., how many clusters per year
        self.age = []
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith('sample_'):
                parts = dir_name.split('_')
                rid, j, age = parts[1], parts[2], parts[3]
                dir_path = os.path.join(self.data_dir, dir_name)
                data_info.append({
                    'dir_path': dir_path,
                    'rid': rid,
                    'j': int(j),
                    'age': float(age)
                })
                self.age.append(int(float(age)))
        self.K=0
        return data_info
    def update_label_first(self, rid, j, age,age_hat1_, feature):
        """
        Input rid, j, age are vectors of length n, feature is an n*d matrix.
        Sort feature, j, rid according to age size.
        Then perform change point detection on feature using l2norm and pelt.
        Get the number of change points K and their positions.
        """
        # Step 1: Sort rid, j, age, feature by age
        sorted_indices = np.argsort(age_hat1_)
        rid_sorted = np.array(rid)[sorted_indices]
        j_sorted = np.array(j)[sorted_indices]
        age_sorted=np.array(age)[sorted_indices]
        feature_sorted = feature[sorted_indices]

        # Step 2: Perform change point detection on sorted features
        model = rpt.Pelt(model="l2", min_size=3).fit(feature)

        # Use predict method for change point detection, return breakpoint list

        # Initialize pen, gradually adjust pen to find approximately 40 intervals
        pen = 1
        detected_breakpoints = []
        while len(detected_breakpoints) < 30- 1: # number of intervals = number of breakpoints + 1
            detected_breakpoints = model.predict(pen=pen)
            pen /= 1.1  # if too few breakpoints, decrease pen
        while len(detected_breakpoints) > 40: # number of intervals = number of breakpoints + 1
            detected_breakpoints = model.predict(pen=pen)
            pen *= 1.1  # if too many breakpoints, increase pen

        # Step 3: Calculate the number and positions of change points
        K = len(detected_breakpoints)   # 变点个数为区间数 - 1
        # breakpoints = detected_breakpoints[:-1]  # 最后一个是结尾位置，去掉它
        segment_means = np.zeros((K, feature_sorted.shape[1]))  # K*d的nparray
        start = 0
        cluster_labels = np.zeros(len(age_sorted), dtype=int)

        for i, bp in enumerate(detected_breakpoints):
            segment_means[i, :] = feature_sorted[start:bp].mean(axis=0)

            cluster_labels[start:bp] = i  # 将当前区间内的样本标记为所属的簇
            start = bp
        # 返回排序后的rid, j, clinical_age, feature，以及变点个数和位置,更新dataset
        self.K=K
        self.cluster_means = segment_means
        data_info = []
        for i in range(len(j_sorted )):
            dir_name='sample_'+str(rid_sorted[i])+'_'+str(j_sorted[i])+'_'+str(age_sorted[i])
            dir_path = os.path.join(self.data_dir, dir_name)
            data_info.append({
                'dir_path': dir_path,
                'rid': rid_sorted[i],
                'j': int(j_sorted[i]),
                'age': float(age_sorted[i])
            })
        self.age=cluster_labels
        self.data_info=data_info

        return  K,  detected_breakpoints

    def update_label(self, rid, j, age, feature):
        """此时已是第一个epoch训练完毕，即我们已知每个cluster的特征是多少。"""
        # 初始化存储结果的列表
        cluster_assignments = []
        rid_new=[]
        j_new=[]
        age_new=[]
        feature_new=[]
        for rid_value in set(rid):  # 针对每个rid分别求解一个优化模型
            # 取出相应rid的子集索引
            subset_index_i = [index for index, value in enumerate(rid) if value == rid_value]
            feature_sub = feature[subset_index_i]
            j_sub = np.array(j)[subset_index_i]
            age_sub = np.array(age)[subset_index_i]

            # 按照age对子集中的j, age, feature进行排序
            sorted_indices = np.argsort(age_sub)
            j_sorted = j_sub[sorted_indices]
            age_sorted = age_sub[sorted_indices]
            feature_sorted = feature_sub[sorted_indices]

            best_cluster = None
            best_loss = float('inf')

            for k in range(self.K):
                loss = 0  # 计算其第一张图assign到k时的初始loss
                pre_age = age_sorted[0]

                for idx in range(len(age_sorted)):
                    gap =  age_sorted[idx]-pre_age
                    adjusted_cluster = int(k + gap * self.precision)
                    if adjusted_cluster>self.K-1:
                        loss = 999999
                        break
                    loss += F.mse_loss(torch.tensor(self.cluster_means[adjusted_cluster]),
                                       torch.tensor(feature_sorted[idx]))
                    # pre_age = age_sorted[idx]

                # 如果当前簇的loss小于之前计算的最优loss，则更新最优簇分配
                if loss < best_loss:
                    best_loss = loss
                    best_cluster = k

            # 存储该rid子集的最优簇分配

            for idx in range(len(age_sorted)):#把新分配的cluster label 存储起来
                rid_new.append(rid_value)
                j_new.append(j_sorted[idx])
                age_new.append(age_sorted[idx])
                gap = age_sorted[idx] - pre_age
                cluster_assignments.append( int(best_cluster+ gap * self.precision))
                feature_new.append(feature_sorted[idx])
        feature_new=np.vstack(feature_new)
        segment_means = np.zeros((self.K, feature_new.shape[1]))  # K*d的nparray
        for i in range(self.K):
            if sum(np.array(cluster_assignments)==i)>0:
                self.cluster_means[i, :] = feature_new[np.array(cluster_assignments)==i].mean(axis=0)
        self.segment_means = segment_means
        self.rid=rid_new
        self.j=j_new
        self.age=cluster_assignments
        data_info=[]
        for i in range(len(j_new )):
            dir_name='sample_'+str(rid_new[i])+'_'+str(j_new[i])+'_'+str(age_new[i])
            dir_path = os.path.join(self.data_dir, dir_name)
            data_info.append({
                'dir_path': dir_path,
                'rid': rid_new[i],
                'j': int(j_new[i]),
                'age': float(age_new[i])
            })
        return cluster_assignments


    def __len__(self):
        # 返回不同 rid 和 j 的组合的数量
        return len(self.data_info)

    def __getitem__(self, index,stage=0):#img2是相同rid但不同j，img3是不同的rid
        # 获取当前index对应的rid和j组合
        selected_data = self.data_info[index]

        # 从0-9随机选择一个数i
        i = random.randint(0, 19)

        # 获取该rid和j组合下的图像
        selected_image = f"{i}.pkl"
        with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
            img1_1 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img1_2 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img1_3 = pickle.load(f)['X']
        # 拼接三通道图像
        img1 = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_2).unsqueeze(0), torch.Tensor(img1_3).unsqueeze(0)], dim=0)
        age1 = selected_data['age']
        clinical_age1 = self.age[index]

        # 找到相同rid但不同j的所有索引
        same_rid_different_j_indices = [
            idx for idx, info in enumerate(self.data_info)
            if info['rid'] == selected_data['rid'] and info['j'] != selected_data['j']
        ]

        # 如果存在不同j的情况，随机选择一个索引
        if same_rid_different_j_indices:
            different_j_index = random.choice(same_rid_different_j_indices)
            selected_image_info_j = self.data_info[different_j_index]
            selected_image_j = f"{i}.pkl"
            with open(os.path.join(selected_image_info_j['dir_path'], selected_image_j), 'rb') as f:
                img2_1 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+10}.pkl"), 'rb') as f:
                img2_2 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+20}.pkl"), 'rb') as f:
                img2_3 = pickle.load(f)['X']
            img2 = torch.cat([torch.Tensor(img2_1).unsqueeze(0), torch.Tensor(img2_2).unsqueeze(0), torch.Tensor(img2_3).unsqueeze(0)], dim=0)
            age2 = selected_image_info_j['age']
        else:
            img2 = img1  # 如果没有不同j的情况，可以选择返回相同的图像
            age2 = age1
            different_j_index = index
        clinical_age2 = self.age[different_j_index]

        # 随机选择一个不同rid的索引
        different_rid_index = index
        while True:
            different_rid_index = random.randint(0, len(self.data_info) - 1)
            if self.data_info[different_rid_index]['rid'] != selected_data['rid']:
                break

        # 使用不同rid的索引获取图像
        same_age_different_rid_indices = [
            i for i, info in enumerate(self.data_info)
            if self.age[i] == self.age[index] and info['rid'] != selected_data['rid']
        ]
        if same_age_different_rid_indices:
            different_rid_index = random.choice(same_age_different_rid_indices)
        else:
            same_age_different_rid_indices = [
                i for i, info in enumerate(self.data_info)
                if abs(self.age[i] - self.age[index]) < 2 and info['rid'] != selected_data['rid']
            ]

        selected_image_info_rid = self.data_info[different_rid_index]
        selected_image_rid = f"{i}.pkl"
        with open(os.path.join(selected_image_info_rid['dir_path'], selected_image_rid), 'rb') as f:
            img3_1 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img3_2 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img3_3 = pickle.load(f)['X']
        img3 = torch.cat([torch.Tensor(img3_1).unsqueeze(0), torch.Tensor(img3_2).unsqueeze(0), torch.Tensor(img3_3).unsqueeze(0)], dim=0)
        age3 = selected_image_info_rid['age']
        clinical_age3 = self.age[different_rid_index]

        # Apply transformations and normalize
        img1 = self.tensor_transforms(img1) / 256.0
        img2 = self.tensor_transforms(img2) / 256.0
        img3 = self.tensor_transforms(img3) / 256.0


        return img1.float(), age1, index, img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index,clinical_age1,clinical_age2,clinical_age3,self.cluster_means[clinical_age1],self.cluster_means[clinical_age2],self.cluster_means[clinical_age3]

class MRIDataset_inference(Dataset):
    def __init__(self, data_dir, resize_dim, train=False):
        self.data_dir = data_dir
        self.age=[]#this is clinical age
        if resize_dim is None:
            resize_dim = 128
        self.resize_dim = resize_dim
        self.data_info = self._load_data_info()
        self.K=0
        self.cluster_means = np.zeros((999,resize_dim))

        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.Lambda(lambda img: transforms.Pad((0, 3))(img) if img.shape[0] == 121 else img),
                transforms.CenterCrop((128, 128)),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.Lambda(lambda img: transforms.Pad((0, 3))(img) if img.shape[0] == 121 else img),
                transforms.CenterCrop((128, 128)),
            ])
    def _load_data_info(self):
        # Load all combinations of rid and j, and store path information
        data_info = []
        self.precision=2#即一年有几个cluster
        self.age = []
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith('sample_'):
                parts = dir_name.split('_')
                rid, j, age = parts[1], parts[2], parts[3]
                dir_path = os.path.join(self.data_dir, dir_name)
                data_info.append({
                    'dir_path': dir_path,
                    'rid': rid,
                    'j': int(j),
                    'age': float(age)
                })
                self.age.append(int(float(age)))
        self.K=0
        return data_info

    def __len__(self):
        # 返回不同 rid 和 j 的组合的数量
        return len(self.data_info)

    def __getitem__(self, index,stage=0):#img2是相同rid但不同j，img3是不同的rid
        # 获取当前index对应的rid和j组合
        selected_data = self.data_info[index]

        # 获取该rid和j组合下的所有图像
        selected_data = self.data_info[index]
        i = 5

        # 获取该rid和j组合下的图像
        selected_image = f"{i}.pkl"
        with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
            img1_1 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+5}.pkl"), 'rb') as f:
            img1_11 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img1_2 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+15}.pkl"), 'rb') as f:
            img1_21 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img1_3 = pickle.load(f)['X']
        # 拼接三通道图像
        img1 = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_11).unsqueeze(0),  torch.Tensor(img1_2).unsqueeze(0),  torch.Tensor(img1_21).unsqueeze(0), torch.Tensor(img1_3).unsqueeze(0)], dim=0)
        age1 = selected_data['age']
        rid=selected_data['rid']
        clinical_age1=self.age[index]
        j=selected_data['j']
        # 找到相同rid但不同j的所有索引

        img1 = self.tensor_transforms(img1) / 1.5

        return img1.float(), age1, index,clinical_age1,self.cluster_means[clinical_age1],rid,j
class MRIDataset_noalign(Dataset):
    def __init__(self, data_dir, resize_dim, train=False):
        self.data_dir = data_dir
        self.age=[]#this is clinical age
        if resize_dim is None:
            resize_dim = 256
        self.resize_dim = resize_dim
        self.precision = 1
        self.data_info = self._load_data_info()
        self.K=0
        self.cluster_means = np.zeros((999,resize_dim))

        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=self.resize_dim, scale=(0.9, 1.1), ratio=(1.0, 1.0),antialias=True),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize_dim,antialias=True),
            ])



    def _load_data_info(self):
        # Load all combinations of rid and j, and store path information
        data_info = []
        self.precision=1#i.e., how many clusters per year
        self.age = []
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith('sample_'):
                parts = dir_name.split('_')
                rid, j, age = parts[1], parts[2], parts[3]
                dir_path = os.path.join(self.data_dir, dir_name)
                data_info.append({
                    'dir_path': dir_path,
                    'rid': rid,
                    'j': int(j),
                    'age': float(age)
                })
                self.age.append(int(float(age)))
        self.K=0
        return data_info


    def __len__(self):
        # 返回不同 rid 和 j 的组合的数量
        return len(self.data_info)

    def __getitem__(self, index,stage=0):#img2是相同rid但不同j，img3是不同的rid
        # 获取当前index对应的rid和j组合
        selected_data = self.data_info[index]

        # 获取该rid和j组合下的所有图像
        images = os.listdir(selected_data['dir_path'])
        selected_image = random.choice(images)
        with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
            img1 = pickle.load(f)['X']
        age1 = selected_data['age']
        clinical_age1=self.age[index]
        # 找到相同rid但不同j的所有索引
        same_rid_different_j_indices = [
            idx for idx, info in enumerate(self.data_info)
            if info['rid'] == selected_data['rid'] and info['j'] != selected_data['j']
        ]

        # 如果存在不同j的情况，随机选择一个索引
        if same_rid_different_j_indices:
            different_j_index = random.choice(same_rid_different_j_indices)
            selected_image_info_j = self.data_info[different_j_index]
            images_j = os.listdir(selected_image_info_j['dir_path'])
            selected_image_j = random.choice(images_j)
            with open(os.path.join(selected_image_info_j['dir_path'], selected_image_j), 'rb') as f:
                img2 = pickle.load(f)['X']
            age2 = selected_image_info_j['age']
        else:
            img2 = img1  # 如果没有不同j的情况，可以选择返回相同的图像
            age2 = age1
            different_j_index = index
        clinical_age2 = self.age[different_j_index]
        # 随机选择一个不同rid的索引
        different_rid_index = index
        while True:
            different_rid_index = random.randint(0, len(self.data_info) - 1)
            if self.data_info[different_rid_index]['rid'] != selected_data['rid']:
                break

        # 使用不同rid的索引获取图像
        same_age_different_rid_indices = [
            i for i, info in enumerate(self.data_info)
            if self.age[i] == self.age[index] and info['rid'] != selected_data['rid']
        ]
        if same_age_different_rid_indices:
        # 随机选择一个符合条件的索引
            different_rid_index = random.choice(same_age_different_rid_indices)
        else:
            same_age_different_rid_indices = [
                i for i, info in enumerate(self.data_info)
                if abs(self.age[i] - self.age[index])<2 and info['rid'] != selected_data['rid']
            ]

        # 使用不同rid的索引获取图像
        selected_image_info_rid = self.data_info[different_rid_index]
        images_rid = os.listdir(selected_image_info_rid['dir_path'])
        selected_image_rid = random.choice(images_rid)
        with open(os.path.join(selected_image_info_rid['dir_path'], selected_image_rid), 'rb') as f:
            img3 = pickle.load(f)['X']
        age3 = selected_image_info_rid['age']
        clinical_age3 = self.age[different_rid_index]

        img2 = self.tensor_transforms(img2)/256
        img3 = self.tensor_transforms(img3)/256
        img1 = self.tensor_transforms(img1) / 256

        return img1.float(), age1, index, img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index,clinical_age1,clinical_age2,clinical_age3,self.cluster_means[clinical_age1],self.cluster_means[clinical_age2],self.cluster_means[clinical_age3]
class MRIDataset_noalign(Dataset):
    def __init__(self, data_dir, resize_dim, train=False):
        self.data_dir = data_dir
        self.age=[]#this is clinical age
        if resize_dim is None:
            resize_dim = 256
        self.resize_dim = resize_dim
        self.precision = 1
        self.data_info = self._load_data_info()
        self.K=0
        self.cluster_means = np.zeros((999,resize_dim))

        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=self.resize_dim, scale=(0.9, 1.1), ratio=(1.0, 1.0),antialias=True),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize_dim,antialias=True),
            ])



    def _load_data_info(self):
        # Load all combinations of rid and j, and store path information
        data_info = []
        self.precision=1#i.e., how many clusters per year
        self.age = []
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith('sample_'):
                parts = dir_name.split('_')
                rid, j, age = parts[1], parts[2], parts[3]
                dir_path = os.path.join(self.data_dir, dir_name)
                data_info.append({
                    'dir_path': dir_path,
                    'rid': rid,
                    'j': int(j),
                    'age': float(age)
                })
                self.age.append(int(float(age)))
        self.K=0
        return data_info


    def __len__(self):
        # 返回不同 rid 和 j 的组合的数量
        return len(self.data_info)

    def __getitem__(self, index,stage=0):#img2是相同rid但不同j，img3是不同的rid
        # 获取当前index对应的rid和j组合
        selected_data = self.data_info[index]

        # 获取该rid和j组合下的所有图像
        images = os.listdir(selected_data['dir_path'])
        selected_image = random.choice(images)
        with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
            img1 = pickle.load(f)['X']
        age1 = selected_data['age']
        clinical_age1=self.age[index]
        # 找到相同rid但不同j的所有索引
        same_rid_different_j_indices = [
            idx for idx, info in enumerate(self.data_info)
            if info['rid'] == selected_data['rid'] and info['j'] != selected_data['j']
        ]

        # 如果存在不同j的情况，随机选择一个索引
        if same_rid_different_j_indices:
            different_j_index = random.choice(same_rid_different_j_indices)
            selected_image_info_j = self.data_info[different_j_index]
            images_j = os.listdir(selected_image_info_j['dir_path'])
            selected_image_j = random.choice(images_j)
            with open(os.path.join(selected_image_info_j['dir_path'], selected_image_j), 'rb') as f:
                img2 = pickle.load(f)['X']
            age2 = selected_image_info_j['age']
        else:
            img2 = img1  # 如果没有不同j的情况，可以选择返回相同的图像
            age2 = age1
            different_j_index = index
        clinical_age2 = self.age[different_j_index]
        # 随机选择一个不同rid的索引
        different_rid_index = index
        while True:
            different_rid_index = random.randint(0, len(self.data_info) - 1)
            if self.data_info[different_rid_index]['rid'] != selected_data['rid']:
                break

        # 使用不同rid的索引获取图像
        same_age_different_rid_indices = [
            i for i, info in enumerate(self.data_info)
            if self.age[i] == self.age[index] and info['rid'] != selected_data['rid']
        ]
        if same_age_different_rid_indices:
        # 随机选择一个符合条件的索引
            different_rid_index = random.choice(same_age_different_rid_indices)
        else:
            same_age_different_rid_indices = [
                i for i, info in enumerate(self.data_info)
                if abs(self.age[i] - self.age[index])<2 and info['rid'] != selected_data['rid']
            ]

        # 使用不同rid的索引获取图像
        selected_image_info_rid = self.data_info[different_rid_index]
        images_rid = os.listdir(selected_image_info_rid['dir_path'])
        selected_image_rid = random.choice(images_rid)
        with open(os.path.join(selected_image_info_rid['dir_path'], selected_image_rid), 'rb') as f:
            img3 = pickle.load(f)['X']
        age3 = selected_image_info_rid['age']
        clinical_age3 = self.age[different_rid_index]

        img2 = self.tensor_transforms(img2)/256
        img3 = self.tensor_transforms(img3)/256
        img1 = self.tensor_transforms(img1) / 256

        return img1.float(), age1, index, img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index,clinical_age1,clinical_age2,clinical_age3,self.cluster_means[clinical_age1],self.cluster_means[clinical_age2],self.cluster_means[clinical_age3]
class MRIDataset_noalign_three(Dataset):
    def __init__(self, data_dir, resize_dim, train=False,with_roi=False):
        self.data_dir = data_dir
        self.age=[]#this is clinical age
        if resize_dim is None:
            resize_dim = 128
        self.resize_dim = resize_dim
        self.precision = 1
        self.data_info = self._load_data_info()
        self.K=0
        self.with_roi=with_roi
        self.cluster_means = np.zeros((999,64))
        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.Lambda(lambda img: transforms.Pad((0, 3))(img) if img.shape[0] == 121 else img),
                transforms.CenterCrop((128, 128)),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.Lambda(lambda img: transforms.Pad((0, 3))(img) if img.shape[0] == 121 else img),
                transforms.CenterCrop((128, 128)),
            ])



    def _load_data_info(self):
        # Load all combinations of rid and j, and store path information
        data_info = []
        self.precision=1#i.e., how many clusters per year
        self.age = []
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith('sample_'):
                parts = dir_name.split('_')
                rid, j, age = parts[1], parts[2], parts[3]
                dir_path = os.path.join(self.data_dir, dir_name)
                data_info.append({
                    'dir_path': dir_path,
                    'rid': rid,
                    'j': int(j),
                    'age': float(age)
                })
                self.age.append(int(float(age)))
        self.K=0
        return data_info
    def update_label_first(self, rid, j, age,age_hat1_, feature):
        """
        Input rid, j, age are vectors of length n, feature is an n*d matrix.
        Sort feature, j, rid according to age size.
        Then perform change point detection on feature using l2norm and pelt.
        Get the number of change points K and their positions.
        """
        # Step 1: Sort rid, j, age, feature by age
        sorted_indices = np.argsort(age_hat1_)
        rid_sorted = np.array(rid)[sorted_indices]
        j_sorted = np.array(j)[sorted_indices]
        age_sorted=np.array(age)[sorted_indices]
        feature_sorted = feature[sorted_indices]

        # Step 2: Perform change point detection on sorted features
        model = rpt.Pelt(model="l2", min_size=3).fit(feature)

        # Use predict method for change point detection, return breakpoint list

        # Initialize pen, gradually adjust pen to find approximately 40 intervals
        pen = 1
        detected_breakpoints = []
        while len(detected_breakpoints) < 20- 1: # 因为区间数 = 断点数 + 1
            detected_breakpoints = model.predict(pen=pen)
            pen /= 1.1  # if too few breakpoints, decrease pen
        while len(detected_breakpoints) > 30: # 因为区间数 = 断点数 + 1
            detected_breakpoints = model.predict(pen=pen)
            pen *= 1.1  # if too many breakpoints, increase pen

        # Step 3: Calculate the number and positions of change points
        K = len(detected_breakpoints)   # 变点个数为区间数 - 1
        # breakpoints = detected_breakpoints[:-1]  # 最后一个是结尾位置，去掉它
        segment_means = np.zeros((K, feature_sorted.shape[1]))  # K*d的nparray
        start = 0
        cluster_labels = np.zeros(len(age_sorted), dtype=int)
        length=[]
        for i, bp in enumerate(detected_breakpoints):
            segment_means[i, :] = feature_sorted[start:bp].mean(axis=0)
            length.append(bp-start)
            cluster_labels[start:bp] = i  # 将当前区间内的样本标记为所属的簇
            start = bp
        # 返回排序后的rid, j, clinical_age, feature，以及变点个数和位置,更新dataset
        self.K=K
        self.cluster_means = segment_means
        data_info = []
        for i in range(len(j_sorted )):
            dir_name='sample_'+str(rid_sorted[i])+'_'+str(j_sorted[i])+'_'+str(age_sorted[i])
            dir_path = os.path.join(self.data_dir, dir_name)
            data_info.append({
                'dir_path': dir_path,
                'rid': rid_sorted[i],
                'j': int(j_sorted[i]),
                'age': float(age_sorted[i])
            })
        self.age=cluster_labels
        self.data_info=data_info
        state="cluster"
        wandb.log({state+ "/num": K, state+"/breakpoints":detected_breakpoints, state+"/max":np.max(length), state+"/min":np.min(length),state+"/mean":np.mean(length)})
        return  K,  detected_breakpoints
    # def update_centriod(self, age, feature):
    #     self.cluster_means[i,:]=np.mean(feature[index,:]) index是age=i对应的角标子集
    #     return

    def __len__(self):
        # 返回不同 rid 和 j 的组合的数量
        return len(self.data_info)

    def __getitem__(self, index, stage=0):
        # 获取当前index对应的rid和j组合
        selected_data = self.data_info[index]

        # 从0-9随机选择一个数i
        i = random.randint(0, 19)

        # 获取该rid和j组合下的图像
        selected_image = f"{i}.pkl"
        with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
            img1_1 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+5}.pkl"), 'rb') as f:
            img1_11 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img1_2 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+15}.pkl"), 'rb') as f:
            img1_21 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img1_3 = pickle.load(f)['X']
        # 拼接三通道图像
        img1 = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_11).unsqueeze(0),  torch.Tensor(img1_2).unsqueeze(0),  torch.Tensor(img1_21).unsqueeze(0), torch.Tensor(img1_3).unsqueeze(0)], dim=0)
        age1 = selected_data['age']
        clinical_age1 = self.age[index]

        # 找到相同rid但不同j的所有索引
        same_rid_different_j_indices = [
            idx for idx, info in enumerate(self.data_info)
            if info['rid'] == selected_data['rid'] and info['j'] != selected_data['j']
        ]

        # 如果存在不同j的情况，随机选择一个索引
        if same_rid_different_j_indices:
            different_j_index = random.choice(same_rid_different_j_indices)
            selected_image_info_j = self.data_info[different_j_index]
            selected_image_j = f"{i}.pkl"
            with open(os.path.join(selected_image_info_j['dir_path'], selected_image_j), 'rb') as f:
                img2_1 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+5}.pkl"), 'rb') as f:
                img2_11 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+10}.pkl"), 'rb') as f:
                img2_2 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+15}.pkl"), 'rb') as f:
                img2_21 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+20}.pkl"), 'rb') as f:
                img2_3 = pickle.load(f)['X']
            img2 = torch.cat([torch.Tensor(img2_1).unsqueeze(0), torch.Tensor(img2_11).unsqueeze(0),torch.Tensor(img2_2).unsqueeze(0), torch.Tensor(img2_21).unsqueeze(0), torch.Tensor(img2_3).unsqueeze(0)], dim=0)
            age2 = selected_image_info_j['age']
        else:
            img2 = img1  # 如果没有不同j的情况，可以选择返回相同的图像
            age2 = age1
            different_j_index = index
        clinical_age2 = self.age[different_j_index]

        # 随机选择一个不同rid的索引
        different_rid_index = index
        while True:
            different_rid_index = random.randint(0, len(self.data_info) - 1)
            if self.data_info[different_rid_index]['rid'] != selected_data['rid']:
                break

        # 使用不同rid的索引获取图像
        same_age_different_rid_indices = [
            i for i, info in enumerate(self.data_info)
            if self.age[i] == self.age[index] and info['rid'] != selected_data['rid']
        ]
        if same_age_different_rid_indices:
            different_rid_index = random.choice(same_age_different_rid_indices)
        else:
            same_age_different_rid_indices = [
                i for i, info in enumerate(self.data_info)
                if abs(self.age[i] - self.age[index]) < 2 and info['rid'] != selected_data['rid']
            ]

        selected_image_info_rid = self.data_info[different_rid_index]
        selected_image_rid = f"{i}.pkl"
        with open(os.path.join(selected_image_info_rid['dir_path'], selected_image_rid), 'rb') as f:
            img3_1 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i + 5}.pkl"), 'rb') as f:
            img3_11 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img3_2 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+15}.pkl"), 'rb') as f:
            img3_21 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img3_3 = pickle.load(f)['X']
        img3 = torch.cat([torch.Tensor(img3_1).unsqueeze(0), torch.Tensor(img3_11).unsqueeze(0), torch.Tensor(img3_2).unsqueeze(0), torch.Tensor(img3_21).unsqueeze(0), torch.Tensor(img3_3).unsqueeze(0)], dim=0)
        age3 = selected_image_info_rid['age']
        clinical_age3 = self.age[different_rid_index]

        # Apply transformations and normalize
        img1 = self.tensor_transforms(img1) / 1.5
        img2 = self.tensor_transforms(img2) / 1.5
        img3 = self.tensor_transforms(img3) / 1.5
        if self.with_roi:
            selected_image = f"{i}.pkl"
            with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
                img1_1 = pickle.load(f)['ROI']
            with open(os.path.join(selected_data['dir_path'], f"{i + 10}.pkl"), 'rb') as f:
                img1_2 = pickle.load(f)['ROI']
            with open(os.path.join(selected_data['dir_path'], f"{i + 20}.pkl"), 'rb') as f:
                img1_3 = pickle.load(f)['ROI']
            img1_roi = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_2).unsqueeze(0),
                              torch.Tensor(img1_3).unsqueeze(0)], dim=0)

            return img1.float(), age1, selected_data[
                'rid'], img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index, clinical_age1, clinical_age2, clinical_age3,self.cluster_means[clinical_age1], self.cluster_means[clinical_age2], self.cluster_means[clinical_age3],img1_roi


        return img1.float(), age1, selected_data['rid'] , img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index,clinical_age1,clinical_age2,clinical_age3,self.cluster_means[clinical_age1],self.cluster_means[clinical_age2],self.cluster_means[clinical_age3]
class MRIDataset_noalign_age_vector(Dataset):
    def __init__(self, data_dir, resize_dim, train=False,ind=0,with_roi=False,larger=False):
        self.data_dir = data_dir
        self.age=[]#this is clinical age
        self.larger=larger
        if resize_dim is None:
            resize_dim =128
        self.resize_dim = resize_dim
        self.precision = 1
        self.data_info = self._load_data_info()
        self.K=0
        self.cluster_means = np.zeros((999,resize_dim))
        self.ind=ind
        self.with_roi = with_roi


        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.Lambda(lambda img: transforms.Pad((0, 3))(img) if img.shape[0] == 121 else img),
                transforms.CenterCrop((128, 128)),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.Lambda(lambda img: transforms.Pad((0, 3))(img) if img.shape[0] == 121 else img),
                transforms.CenterCrop((128, 128)),
            ])

    def _load_data_info(self):
        # Load all combinations of rid and j, and store path info
        if self.larger:
            rid_to_sample = {}
            self.age = []
            for dir_name in os.listdir(self.data_dir):
                if dir_name.startswith('sample_'):
                    parts = dir_name.split('_')
                    rid, j, age = parts[1], parts[2], parts[3]
                    dir_path = os.path.join(self.data_dir, dir_name)
                    age = float(age)
                    sample_info = {
                        'dir_path': dir_path,
                        'rid': rid,
                        'j': int(j),
                        'age': age
                    }
                    # Check if this rid is already in the dictionary or if the current age is smaller
                    if rid not in rid_to_sample or age < rid_to_sample[rid]['age']:
                        rid_to_sample[rid] = sample_info

            # Convert the dictionary values to a list for data_info
            data_info = list(rid_to_sample.values())
            self.age = [int(sample['age']) for sample in data_info]
            self.K = 0
            return data_info
        else:
            # Load all combinations of rid and j, and store path information
            data_info = []
            self.precision = 1  # 即一年有几个cluster
            self.age = []
            for dir_name in os.listdir(self.data_dir):
                if dir_name.startswith('sample_'):
                    parts = dir_name.split('_')
                    rid, j, age = parts[1], parts[2], parts[3]
                    dir_path = os.path.join(self.data_dir, dir_name)
                    data_info.append({
                        'dir_path': dir_path,
                        'rid': rid,
                        'j': int(j),
                        'age': float(age)
                    })
                    self.age.append(int(float(age)))
            self.K = 0
            return data_info


    def __len__(self):
        # 返回不同 rid 和 j 的组合的数量
        return len(self.data_info)

    def __getitem__(self, index):
        # 获取当前index对应的rid和j组合
        selected_data = self.data_info[index]
        i =self.ind
        # 获取该rid和j组合下的图像
        selected_image = f"{i}.pkl"
        with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
            img1_1 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+5}.pkl"), 'rb') as f:
            img1_11 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img1_2 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+15}.pkl"), 'rb') as f:
            img1_21 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img1_3 = pickle.load(f)['X']
        # 拼接三通道图像
        img1 = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_11).unsqueeze(0),  torch.Tensor(img1_2).unsqueeze(0),  torch.Tensor(img1_21).unsqueeze(0), torch.Tensor(img1_3).unsqueeze(0)], dim=0)
        age1 = selected_data['age']
        clinical_age1 = self.age[index]

        # 找到相同rid但不同j的所有索引
        same_rid_different_j_indices = [
            idx for idx, info in enumerate(self.data_info)
            if info['rid'] == selected_data['rid'] and info['j'] != selected_data['j']
        ]

        # 如果存在不同j的情况，随机选择一个索引
        if same_rid_different_j_indices:
            different_j_index = random.choice(same_rid_different_j_indices)
            selected_image_info_j = self.data_info[different_j_index]
            selected_image_j = f"{i}.pkl"
            with open(os.path.join(selected_image_info_j['dir_path'], selected_image_j), 'rb') as f:
                img2_1 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+5}.pkl"), 'rb') as f:
                img2_11 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+10}.pkl"), 'rb') as f:
                img2_2 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+15}.pkl"), 'rb') as f:
                img2_21 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+20}.pkl"), 'rb') as f:
                img2_3 = pickle.load(f)['X']
            img2 = torch.cat([torch.Tensor(img2_1).unsqueeze(0), torch.Tensor(img2_11).unsqueeze(0),torch.Tensor(img2_2).unsqueeze(0), torch.Tensor(img2_21).unsqueeze(0), torch.Tensor(img2_3).unsqueeze(0)], dim=0)
            age2 = selected_image_info_j['age']
        else:
            img2 = img1  # 如果没有不同j的情况，可以选择返回相同的图像
            age2 = age1
            different_j_index = index
        clinical_age2 = self.age[different_j_index]

        # 随机选择一个不同rid的索引
        different_rid_index = index
        while True:
            different_rid_index = random.randint(0, len(self.data_info) - 1)
            if self.data_info[different_rid_index]['rid'] != selected_data['rid']:
                break

        # 使用不同rid的索引获取图像
        same_age_different_rid_indices = [
            i for i, info in enumerate(self.data_info)
            if self.age[i] == self.age[index] and info['rid'] != selected_data['rid']
        ]
        if same_age_different_rid_indices:
            different_rid_index = random.choice(same_age_different_rid_indices)
        else:
            same_age_different_rid_indices = [
                i for i, info in enumerate(self.data_info)
                if abs(self.age[i] - self.age[index]) < 2 and info['rid'] != selected_data['rid']
            ]

        selected_image_info_rid = self.data_info[different_rid_index]
        selected_image_rid = f"{i}.pkl"
        with open(os.path.join(selected_image_info_rid['dir_path'], selected_image_rid), 'rb') as f:
            img3_1 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i + 5}.pkl"), 'rb') as f:
            img3_11 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img3_2 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+15}.pkl"), 'rb') as f:
            img3_21 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img3_3 = pickle.load(f)['X']
        img3 = torch.cat([torch.Tensor(img3_1).unsqueeze(0), torch.Tensor(img3_11).unsqueeze(0), torch.Tensor(img3_2).unsqueeze(0), torch.Tensor(img3_21).unsqueeze(0), torch.Tensor(img3_3).unsqueeze(0)], dim=0)
        age3 = selected_image_info_rid['age']
        clinical_age3 = self.age[different_rid_index]

        # Apply transformations and normalize
        img1 = self.tensor_transforms(img1) / 1.5
        img2 = self.tensor_transforms(img2) / 1.5
        img3 = self.tensor_transforms(img3) / 1.5
        if self.with_roi:
            selected_image = f"{i}.pkl"
            with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
                img1_1 = pickle.load(f)['ROI']
            with open(os.path.join(selected_data['dir_path'], f"{i + 10}.pkl"), 'rb') as f:
                img1_2 = pickle.load(f)['ROI']
            with open(os.path.join(selected_data['dir_path'], f"{i + 20}.pkl"), 'rb') as f:
                img1_3 = pickle.load(f)['ROI']
            img1_roi = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_2).unsqueeze(0),
                              torch.Tensor(img1_3).unsqueeze(0)], dim=0)

            return img1.float(), age1, selected_data[
                'rid'], img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index, clinical_age1, clinical_age2, clinical_age3, self.cluster_means[clinical_age1], self.cluster_means[clinical_age2], self.cluster_means[clinical_age3],img1_roi

        return img1.float(), age1, selected_data['rid'] , img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index,clinical_age1,clinical_age2,clinical_age3,self.cluster_means[clinical_age1],self.cluster_means[clinical_age2],self.cluster_means[clinical_age3]
class MRIDataset_cross_label(Dataset):
    def __init__(self, data_dir, resize_dim=None, train=False, with_roi=False, split=True):
        self.data_dir = data_dir
        self.age = []  # 这是临床年龄
        if resize_dim is None:
            resize_dim = 256
        self.resize_dim = resize_dim
        self.precision = 1
        self.data_info = self._load_data_info()
        self.K = 0
        self.with_roi = with_roi
        self.cluster_means = np.zeros((999, resize_dim))
        self.split = split

        if train:
            self.tensor_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=self.resize_dim, scale=(0.9, 1.1), ratio=(1.0, 1.0)),
            ])
        else:
            self.tensor_transforms = transforms.Compose([
                transforms.Resize(self.resize_dim),
            ])

        # 随机取出data_info中一半的rid对应的index
        all_rids = list(set([info['rid'] for info in self.data_info]))
        num_rids = len(all_rids)
        num_selected_rids = num_rids // 2
        selected_rids = random.sample(all_rids, num_selected_rids)

        # 获取selected_rids对应的索引
        self.index_subset = [i for i, info in enumerate(self.data_info) if info['rid'] in selected_rids]

        # 初始化data_info_subset和age_subset
        self.update_partition(self.split)

    def _load_data_info(self):
        # Load all combinations of rid and j, and store path information
        data_info = []
        self.precision = 1  # 即一年有几个cluster
        self.age = []
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith('sample_'):
                parts = dir_name.split('_')
                if len(parts) >= 4:
                    rid, j, age = parts[1], parts[2], parts[3]
                    dir_path = os.path.join(self.data_dir, dir_name)
                    data_info.append({
                        'dir_path': dir_path,
                        'rid': rid,
                        'j': int(j),
                        'age': float(age)
                    })
                    self.age.append(int(float(age)))
        self.K = 0
        return data_info

    def update_partition(self, split):
        self.split = split
        if split == True:
            self.data_info_subset = [self.data_info[i] for i in self.index_subset]
            self.age_subset = [self.age[i] for i in self.index_subset]
        else:
            all_indices = list(range(len(self.data_info)))
            other_indices = [i for i in all_indices if i not in self.index_subset]
            self.data_info_subset = [self.data_info[i] for i in other_indices]
            self.age_subset = [self.age[i] for i in other_indices]

    def __len__(self):
        # 返回data_info_subset的长度
        return len(self.data_info_subset)


    def __getitem__(self, index, stage=0):

        # 获取当前index对应的rid和j组合
        selected_data = self.data_info_subset[index]

        # 从0-9随机选择一个数i
        i = random.randint(0, 19)

        # 获取该rid和j组合下的图像
        selected_image = f"{i}.pkl"
        with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
            img1_1 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img1_2 = pickle.load(f)['X']
        with open(os.path.join(selected_data['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img1_3 = pickle.load(f)['X']
        # 拼接三通道图像
        img1 = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_2).unsqueeze(0), torch.Tensor(img1_3).unsqueeze(0)], dim=0)
        age1 = selected_data['age']
        clinical_age1 = self.age_subset[index]

        # 找到相同rid但不同j的所有索引
        same_rid_different_j_indices = [
            idx for idx, info in enumerate(self.data_info_subset)
            if info['rid'] == selected_data['rid'] and info['j'] != selected_data['j']
        ]

        # 如果存在不同j的情况，随机选择一个索引
        if same_rid_different_j_indices:
            different_j_index = random.choice(same_rid_different_j_indices)
            selected_image_info_j = self.data_info_subset[different_j_index]
            selected_image_j = f"{i}.pkl"
            with open(os.path.join(selected_image_info_j['dir_path'], selected_image_j), 'rb') as f:
                img2_1 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+10}.pkl"), 'rb') as f:
                img2_2 = pickle.load(f)['X']
            with open(os.path.join(selected_image_info_j['dir_path'], f"{i+20}.pkl"), 'rb') as f:
                img2_3 = pickle.load(f)['X']
            img2 = torch.cat([torch.Tensor(img2_1).unsqueeze(0), torch.Tensor(img2_2).unsqueeze(0), torch.Tensor(img2_3).unsqueeze(0)], dim=0)
            age2 = selected_image_info_j['age']
        else:
            img2 = img1  # 如果没有不同j的情况，可以选择返回相同的图像
            age2 = age1
            different_j_index = index
        clinical_age2 = self.age_subset[different_j_index]

        # 随机选择一个不同rid的索引
        different_rid_index = index
        while True:
            different_rid_index = random.randint(0, len(self.data_info_subset) - 1)
            if self.data_info_subset[different_rid_index]['rid'] != selected_data['rid']:
                break

        # 使用不同rid的索引获取图像
        same_age_different_rid_indices = [
            i for i, info in enumerate(self.data_info_subset)
            if self.age_subset[i] == self.age_subset[index] and info['rid'] != selected_data['rid']
        ]
        if same_age_different_rid_indices:
            different_rid_index = random.choice(same_age_different_rid_indices)
        else:
            same_age_different_rid_indices = [
                i for i, info in enumerate(self.data_info_subset)
                if abs(self.age_subset[i] - self.age_subset[index]) < 2 and info['rid'] != selected_data['rid']
            ]

        selected_image_info_rid = self.data_info_subset[different_rid_index]
        selected_image_rid = f"{i}.pkl"
        with open(os.path.join(selected_image_info_rid['dir_path'], selected_image_rid), 'rb') as f:
            img3_1 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+10}.pkl"), 'rb') as f:
            img3_2 = pickle.load(f)['X']
        with open(os.path.join(selected_image_info_rid['dir_path'], f"{i+20}.pkl"), 'rb') as f:
            img3_3 = pickle.load(f)['X']
        img3 = torch.cat([torch.Tensor(img3_1).unsqueeze(0), torch.Tensor(img3_2).unsqueeze(0), torch.Tensor(img3_3).unsqueeze(0)], dim=0)
        age3 = selected_image_info_rid['age']
        clinical_age3 = self.age_subset[different_rid_index]

        # Apply transformations and normalize
        img1 = self.tensor_transforms(img1) / 256.0
        img2 = self.tensor_transforms(img2) / 256.0
        img3 = self.tensor_transforms(img3) / 256.0
        if self.with_roi:
            selected_image = f"{i}.pkl"
            with open(os.path.join(selected_data['dir_path'], selected_image), 'rb') as f:
                img1_1 = pickle.load(f)['ROI']
            with open(os.path.join(selected_data['dir_path'], f"{i + 10}.pkl"), 'rb') as f:
                img1_2 = pickle.load(f)['ROI']
            with open(os.path.join(selected_data['dir_path'], f"{i + 20}.pkl"), 'rb') as f:
                img1_3 = pickle.load(f)['ROI']
            img1_roi = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_2).unsqueeze(0),
                              torch.Tensor(img1_3).unsqueeze(0)], dim=0)

            return img1.float(), age1, selected_data[
                'rid'], img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index, clinical_age1, clinical_age2, clinical_age3,self.cluster_means[clinical_age1], self.cluster_means[clinical_age2], self.cluster_means[clinical_age3],img1_roi


        return img1.float(), age1, selected_data['rid'] , img2.float(), age2, different_j_index, img3.float(), age3, different_rid_index,clinical_age1,clinical_age2,clinical_age3,self.cluster_means[clinical_age1],self.cluster_means[clinical_age2],self.cluster_means[clinical_age3]

import torch.nn.functional as F

import torch
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
import math


def downsample_to_128(img_3d: torch.Tensor, rotate_angles=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """
    Downsample a 3D volume from shape (256,256,256) to (128,128,128)
    via optional rotation and trilinear interpolation.

    Args:
        img_3d (torch.Tensor): Input tensor of shape (D,H,W) or (1,D,H,W).
        rotate_angles (tuple): Rotation angles (angle_x, angle_y, angle_z) in radians.

    Returns:
        torch.Tensor: Downsampled tensor of shape (1,128,128,128).
    """
    # Ensure input shape is (1, D, H, W)
    if img_3d.ndim == 3:
        img_3d = img_3d.unsqueeze(0)  # (1, D, H, W)

    img_3d = img_3d.unsqueeze(0)  # (1,1,D,H,W)

    # 如果旋转角度全部为0，则跳过旋转步骤
    if any(abs(angle) > 1e-6 for angle in rotate_angles):
        img_3d = rotate_3d(img_3d, rotate_angles)

    # 裁剪中心区域（移除 18 像素边缘，避免插值误差）
    img_3d = img_3d[:, :, 18:-18, 18:-18, 18:-18]  # (1, 1, 220, 220, 220)

    # 下采样到 (128,128,128)
    out = F.interpolate(img_3d, size=(128, 128, 128), mode='trilinear', align_corners=False)

    return out.squeeze(0)  # (1,128,128,128)


def rotate_3d(img: torch.Tensor, angles=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """
    Apply 3D rotation to a volume using affine grid sampling.

    Args:
        img (torch.Tensor): Input tensor of shape (N, C, D, H, W).
        angles (tuple): Rotation angles (angle_x, angle_y, angle_z) in radians.

    Returns:
        torch.Tensor: Rotated tensor of the same shape.
    """
    angle_x, angle_y, angle_z = angles

    def get_rotation_matrix(angle_x, angle_y, angle_z):
        """Compute 3D rotation matrix (4x4 affine) for given angles."""
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        cos_z, sin_z = math.cos(angle_z), math.sin(angle_z)

        # Rotation matrices
        R_x = torch.tensor([[1, 0, 0, 0],
                            [0, cos_x, -sin_x, 0],
                            [0, sin_x, cos_x, 0],
                            [0, 0, 0, 1]])

        R_y = torch.tensor([[cos_y, 0, sin_y, 0],
                            [0, 1, 0, 0],
                            [-sin_y, 0, cos_y, 0],
                            [0, 0, 0, 1]])

        R_z = torch.tensor([[cos_z, -sin_z, 0, 0],
                            [sin_z, cos_z, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        R = R_z @ R_y @ R_x  # Combined rotation matrix
        return R[:3]  # Extract (3,4) matrix for affine_grid

    # Compute affine rotation matrix
    affine_matrix = get_rotation_matrix(angle_x, angle_y, angle_z).unsqueeze(0).to(img.device).float()  # (1, 3, 4)

    # Generate 3D affine grid
    d, h, w = img.shape[2:]
    grid = F.affine_grid(affine_matrix, size=(1, 1, d, h, w), align_corners=False).to(img.device)

    # Apply 3D rotation
    rotated_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return rotated_img


def random_or_fixed_crop_3d(img_3d: torch.Tensor, train: bool, crop_size=(120,120,120),off_d=4,off_h=4,off_w=4) -> torch.Tensor:
    """
    Crops out a (120,120,120) region from a (1,128,128,128) volume with an offset.
    If train=True, use random offset; else offset=4 (center-like).
    img_3d => shape (1,128,128,128).
    Returns => shape (1,120,120,120).
    """
    # C, D, H, W = img_3d.shape  # C=1, D=128, H=128, W=128
    cd, ch, cw = crop_size     # (120,120,120)



    img_cropped = img_3d[:,
                         off_d:off_d+cd,
                         off_h:off_h+ch,
                         off_w:off_w+cw]
    # if train and random.random() < 0.1:#swap的情况下不能这样，否则random erase也会出现在target image 中
    #     erase_size = 20  # Size of the cube to erase
    #     ed = random.randint(0, cd - erase_size)
    #     eh = random.randint(0, ch - erase_size)
    #     ew = random.randint(0, cw - erase_size)
    #     img_cropped[:, ed:ed + erase_size, eh:eh + erase_size, ew:ew + erase_size] = 0

    return img_cropped  # (1,120,120,120)

class MRIDataset_3D_three(Dataset):
    def __init__(self, data_dir, resize_dim=120, train=False, with_roi=False,mci_ad=False):
        """
        data_dir: directory containing 'sample_*' subfolders.
        Each subfolder has a '3d_mni.pkl' of shape (256,256,256).
        We'll downsample to (128^3), then crop to (120^3).
        """
        super().__init__()
        self.data_dir = data_dir
        self.train = train
        self.resize_dim = resize_dim if resize_dim is not None else 120
        self.with_roi = with_roi
        self.mci_ad = mci_ad
        # Some fields from your original code
        self.age = []
        self.precision = 1
        self.data_info = self._load_data_info()
        self.K = 0
        self.cluster_means = np.zeros((999,64))

        # For mixing logic
        self.patch_size_mri = 8   # => 120/8=15 blocks
        self.patch_size_age = 1   # => 15 blocks for age as well

    def _load_data_info(self):
        data_info = []
        self.precision=1
        self.age=[]
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith('sample_'):
                parts = dir_name.split('_')
                rid, j, age_str = parts[1], parts[2], parts[3]
                dir_path = os.path.join(self.data_dir, dir_name)
                data_info.append({
                    'dir_path': dir_path,
                    'rid': rid,
                    'j': int(j),
                    'age': float(age_str)
                })
                self.age.append(int(float(age_str)))
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        if not self.mci_ad:
            # 1) Current sample data
            selected_data = self.data_info[index]
            rid1 = selected_data['rid']
            j1   = selected_data['j']
            age1 = selected_data['age']
            flip=random.random()>0.5

            # 2) Read 3D volume => shape (256,256,256)
            with open(os.path.join(selected_data['dir_path'], "3d_mni.pkl"), 'rb') as f:
                img1_np = pickle.load(f)['X']
            img1 = torch.tensor(img1_np, dtype=torch.float32)
            if self.train:
                off_d = random.randint(0, 8)
                off_h = random.randint(0, 8)
                off_w = random.randint(0, 8)
                b=random.random()/10+0.95
                if flip:
                    img1=torch.flip(img1, dims=[0])
                    random_rotation_angles = (
                        math.radians(random.uniform(-3, 3)),
                        math.radians(random.uniform(-3, 3)),
                        math.radians(random.uniform(-3, 3))
                    )
                else:
                    random_rotation_angles = (0, 0, 0)
            else:
                flip=False
                # If not training, fix offset to 4 => center-ish
                off_d = off_h = off_w = 4
                b=1
                random_rotation_angles=(0,0,0)

                # 3) Tensor => shape (256,256,256)
            # 4) Downsample => (1,128,128,128)
            img1 = downsample_to_128(img1,random_rotation_angles)

            # 5) Random/Fixed crop => (1,120,120,120)
            img1 = random_or_fixed_crop_3d(img1,self.train,(120,120,120),off_d,off_h,off_w)

            # 6) Normalize => /255
            eps = 1e-8
            img1 = img1 /255*b

            # (If you want a channel dimension of 1 at the front, you can keep it as is:
            # shape => (1,120,120,120). Otherwise you might do unsqueeze(0).)

            # Now do the same for 'img2', 'img3' (like your code) => find different_j_index, etc
            # For brevity, let's show the essential steps only:

            # --- Find another sample index for same rid, different j ---
            same_rid_different_j_indices = [
                idx for idx, info in enumerate(self.data_info)
                if info['rid'] == rid1 and info['j'] != j1
            ]
            if same_rid_different_j_indices:
                different_j_index = random.choice(same_rid_different_j_indices)
            else:
                different_j_index = index

            selected_j_data = self.data_info[different_j_index]
            age2 = selected_j_data['age']
            # Read -> downsample -> crop -> normalize
            with open(os.path.join(selected_j_data['dir_path'], "3d_mni.pkl"), 'rb') as f:
                img2_np = pickle.load(f)['X']
            img2 = torch.tensor(img2_np, dtype=torch.float32)
            if flip:
                img2=torch.flip(img2, dims=[0])
            img2 = downsample_to_128(img2,random_rotation_angles)
            img2 = random_or_fixed_crop_3d(img2, self.train, (120,120,120),off_d,off_h,off_w)
            img2 = img2/255*b

            # Another index for img3
            same_rid_diff_j_indices_2 = [idx for idx in same_rid_different_j_indices if idx!=different_j_index]
            if same_rid_diff_j_indices_2:
                diff_j_index_2 = random.choice(same_rid_diff_j_indices_2)
            else:
                diff_j_index_2 = index

            selected_j2_data = self.data_info[diff_j_index_2]
            age3 = selected_j2_data['age']
            with open(os.path.join(selected_j2_data['dir_path'], "3d_mni.pkl"), 'rb') as f:
                img3_np = pickle.load(f)['X']
            img3 = torch.tensor(img3_np, dtype=torch.float32)
            if flip:
                img3=torch.flip(img3, dims=[0])
            img3 = downsample_to_128(img3,random_rotation_angles)
            img3 = random_or_fixed_crop_3d(img3, self.train, (120,120,120),off_d,off_h,off_w)
            min_val3 = img3.min()
            max_val3 = img3.max()
            img3 = img3  /255*b

            # ============ The patch mixing logic (the same as your original) ============
            # (A) Split img2/img3 into 8x8x8 patches => 15^3=3375 patches
            img2_patches = self._split_into_patches_3d(img2.squeeze(0), patch_size=self.patch_size_mri)
            img3_patches = self._split_into_patches_3d(img3.squeeze(0), patch_size=self.patch_size_mri)

            # (B) Age volumes => shape (15,15,15)
            age2_vol = torch.full((15,15,15), age2, dtype=torch.float32)
            age3_vol = torch.full((15,15,15), age3, dtype=torch.float32)
            age2_patches = self._split_into_patches_3d(age2_vol, patch_size=self.patch_size_age)
            age3_patches = self._split_into_patches_3d(age3_vol, patch_size=self.patch_size_age)

            # (C) Mix
            mixed_img_patches, mixed_age_patches = self._random_mix_patches(
                img2_patches, img3_patches,
                age2_patches, age3_patches
            )

            # (D) Reconstruct => (120,120,120) + (15,15,15)
            mixed_img = self._reconstruct_from_patches_3d(
                mixed_img_patches, full_shape=(120,120,120), patch_size=self.patch_size_mri
            )
            mixed_age= self._reconstruct_from_patches_3d(
                mixed_age_patches, full_shape=(15,15,15), patch_size=self.patch_size_age
            ).unsqueeze(0)
            # Also define age1_vol => shape(15,15,15)
            age1_vol = torch.full((15,15,15), age1, dtype=torch.float32)
        else:

            # 获取当前数据
            selected_data = self.data_info[index]
            rid1 = selected_data['rid']
            j1   = selected_data['j']
            age1 = selected_data['age']

            # 2) Read 3D volume => shape (256,256,256)
            with open(os.path.join(selected_data['dir_path'], "3d_mni.pkl"), 'rb') as f:
                data = pickle.load(f)  # 读取一次
                img1_np = data['X']
                path = data['path']
            if self.train:
                off_d = random.randint(0, 8)
                off_h = random.randint(0, 8)
                off_w = random.randint(0, 8)
                b=random.random()/10+0.95
            else:
                # If not training, fix offset to 4 => center-ish
                off_d = off_h = off_w = 4
                b=1

            # 3) Tensor => shape (256,256,256)
            img1 = torch.tensor(img1_np, dtype=torch.float32)

            # 4) Downsample => (1,128,128,128)
            img1 = downsample_to_128(img1)

            # 5) Random/Fixed crop => (1,120,120,120)
            img1 = random_or_fixed_crop_3d(img1,self.train,(120,120,120),off_d,off_h,off_w)

            # 6) Normalize => /255
            eps = 1e-8
            img1 = img1 /255*b
            # path=selected_data['path']

            # (If you want a channel dimension of 1 at the front, you can keep it as is:
            # shape => (1,120,120,120). Otherwise you might do unsqueeze(0).)

            # Now do the same for 'img2', 'img3' (like your code) => find different_j_index, etc
            # For brevity, let's show the essential steps only:

            # --- Find another sample index for same rid, different j ---
            same_rid_different_j_indices = [
                idx for idx, info in enumerate(self.data_info)
                if info['rid'] == rid1 and info['j'] != j1
            ]
            if same_rid_different_j_indices:
                different_j_index = random.choice(same_rid_different_j_indices)
            else:
                different_j_index = index

            selected_j_data = self.data_info[different_j_index]
            age2 = selected_j_data['age']
            # Read -> downsample -> crop -> normalize
            with open(os.path.join(selected_j_data['dir_path'], "3d_mni.pkl"), 'rb') as f:
                img2_np = pickle.load(f)['X']

            img2 = torch.tensor(img2_np, dtype=torch.float32)
            img2 = downsample_to_128(img2)
            img2 = random_or_fixed_crop_3d(img2, self.train, (120,120,120),off_d,off_h,off_w)
            img2 = img2/255*b

            # Another index for img3
            same_rid_diff_j_indices_2 = [idx for idx in same_rid_different_j_indices if idx!=different_j_index]
            if same_rid_diff_j_indices_2:
                diff_j_index_2 = random.choice(same_rid_diff_j_indices_2)
            else:
                diff_j_index_2 = index

            selected_j2_data = self.data_info[diff_j_index_2]
            age3 = selected_j2_data['age']
            with open(os.path.join(selected_j2_data['dir_path'], "3d_mni.pkl"), 'rb') as f:
                img3_np = pickle.load(f)['X']
            img3 = torch.tensor(img3_np, dtype=torch.float32)
            img3 = downsample_to_128(img3)
            img3 = random_or_fixed_crop_3d(img3, self.train, (120,120,120),off_d,off_h,off_w)
            min_val3 = img3.min()
            max_val3 = img3.max()
            img3 = img3  /255*b

            # ============ The patch mixing logic (the same as your original) ============
            # (A) Split img2/img3 into 8x8x8 patches => 15^3=3375 patches
            img2_patches = self._split_into_patches_3d(img2.squeeze(0), patch_size=self.patch_size_mri)
            img3_patches = self._split_into_patches_3d(img3.squeeze(0), patch_size=self.patch_size_mri)

            # (B) Age volumes => shape (15,15,15)
            age2_vol = torch.full((15,15,15), age2, dtype=torch.float32)
            age3_vol = torch.full((15,15,15), age3, dtype=torch.float32)
            age2_patches = self._split_into_patches_3d(age2_vol, patch_size=self.patch_size_age)
            age3_patches = self._split_into_patches_3d(age3_vol, patch_size=self.patch_size_age)

            # (C) Mix
            mixed_img_patches, mixed_age_patches = self._random_mix_patches(
                img2_patches, img3_patches,
                age2_patches, age3_patches
            )

            # (D) Reconstruct => (120,120,120) + (15,15,15)
            mixed_img = self._reconstruct_from_patches_3d(
                mixed_img_patches, full_shape=(120,120,120), patch_size=self.patch_size_mri
            )
            mixed_age= self._reconstruct_from_patches_3d(
                mixed_age_patches, full_shape=(15,15,15), patch_size=self.patch_size_age
            )
            # Also define age1_vol => shape(15,15,15)
            age1_vol = torch.full((15,15,15), age1, dtype=torch.float32)
        # Return them in the format you prefer
            mixed_img=path
        return (
            img1,                 # shape (1,120,120,120)
            age1_vol,
            rid1,
            img2,
            torch.full((15,15,15), age2, dtype=torch.float32),
            self.data_info[different_j_index]['j'],
            mixed_img,  # add channel => (1,120,120,120)
            mixed_age
        )

    # =============== The rest of your code ===============
    def _split_into_patches_3d(self, tensor_3d, patch_size):
        # same as your original
        D, H, W = tensor_3d.shape
        patches = tensor_3d.unfold(0, patch_size, patch_size) \
                           .unfold(1, patch_size, patch_size) \
                           .unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size, patch_size, patch_size)
        return patches

    def _reconstruct_from_patches_3d(self, patches, full_shape, patch_size):
        # same as your original
        D, H, W = full_shape
        d_blocks = D // patch_size
        h_blocks = H // patch_size
        w_blocks = W // patch_size
        patches = patches.view(d_blocks, h_blocks, w_blocks, patch_size, patch_size, patch_size)
        out_tensor = patches.permute(0,3,1,4,2,5).contiguous().view(D, H, W)
        return out_tensor

    def _random_mix_patches(self, patches2, patches3, ages2, ages3):
        # same logic as your original
        mixed_patches = []
        mixed_ages = []
        for p2, p3, a2, a3 in zip(patches2, patches3, ages2, ages3):
            if random.random() > 0.5:
                mixed_patches.append(p2)
                mixed_ages.append(a2)
            else:
                mixed_patches.append(p3)
                mixed_ages.append(a3)
        return torch.stack(mixed_patches), torch.stack(mixed_ages)