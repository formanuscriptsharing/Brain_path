import os
import pickle
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


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

    # If rotation angles are all 0, skip rotation
    if any(abs(angle) > 1e-6 for angle in rotate_angles):
        img_3d = rotate_3d(img_3d, rotate_angles)

    img_3d = img_3d[:, :, 18:-18, 18:-18, 18:-18]  # (1, 1, 220, 220, 220)

    # Downsample to (128,128,128)
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
    cd, ch, cw = crop_size     # (120,120,120)

    img_cropped = img_3d[:,
                         off_d:off_d+cd,
                         off_h:off_h+ch,
                         off_w:off_w+cw]

    return img_cropped  # (1,120,120,120)


class MRIDataset_inference(Dataset):
    def __init__(self, data_dir, resize_dim, train=False):
        self.data_dir = data_dir
        self.age=[]  # This is clinical age
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
        self.precision=2
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
        return len(self.data_info)

    def __getitem__(self, index,stage=0):
        selected_data = self.data_info[index]
        selected_data = self.data_info[index]
        i = 5

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

        img1 = torch.cat([torch.Tensor(img1_1).unsqueeze(0), torch.Tensor(img1_11).unsqueeze(0),  torch.Tensor(img1_2).unsqueeze(0),  torch.Tensor(img1_21).unsqueeze(0), torch.Tensor(img1_3).unsqueeze(0)], dim=0)
        age1 = selected_data['age']
        rid=selected_data['rid']
        clinical_age1=self.age[index]
        j=selected_data['j']

        img1 = self.tensor_transforms(img1) / 1.5

        return img1.float(), age1, index,clinical_age1,self.cluster_means[clinical_age1],rid,j


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

            # 3) Downsample => (1,128,128,128)
            img1 = downsample_to_128(img1,random_rotation_angles)

            # 4) Random/Fixed crop => (1,120,120,120)
            img1 = random_or_fixed_crop_3d(img1,self.train,(120,120,120),off_d,off_h,off_w)

            # 5) Normalize => /255
            eps = 1e-8
            img1 = img1 /255*b

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

            # ============ The patch mixing logic ============
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
            selected_data = self.data_info[index]
            rid1 = selected_data['rid']
            j1   = selected_data['j']
            age1 = selected_data['age']

            # 2) Read 3D volume => shape (256,256,256)
            with open(os.path.join(selected_data['dir_path'], "3d_mni.pkl"), 'rb') as f:
                data = pickle.load(f)  
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
            j2=selected_j_data['j']

            # Read -> downsample -> crop -> normalize
            with open(os.path.join(selected_j_data['dir_path'], "3d_mni.pkl"), 'rb') as f:
                data = pickle.load(f)
                img2_np = data['X']

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
                data = pickle.load(f)
                img3_np = data['X']

            img3 = torch.tensor(img3_np, dtype=torch.float32)
            img3 = downsample_to_128(img3)
            img3 = random_or_fixed_crop_3d(img3, self.train, (120,120,120),off_d,off_h,off_w)
            min_val3 = img3.min()
            max_val3 = img3.max()
            img3 = img3  /255*b

            # ============ The patch mixing logic ============
            img2_patches = self._split_into_patches_3d(img2.squeeze(0), patch_size=self.patch_size_mri)
            img3_patches = self._split_into_patches_3d(img3.squeeze(0), patch_size=self.patch_size_mri)

            age2_vol = torch.full((15,15,15), age2, dtype=torch.float32)
            age3_vol = torch.full((15,15,15), age3, dtype=torch.float32)
            age2_patches = self._split_into_patches_3d(age2_vol, patch_size=self.patch_size_age)
            age3_patches = self._split_into_patches_3d(age3_vol, patch_size=self.patch_size_age)

            mixed_img_patches, mixed_age_patches = self._random_mix_patches(
                img2_patches, img3_patches,
                age2_patches, age3_patches
            )

            mixed_img = self._reconstruct_from_patches_3d(
                mixed_img_patches, full_shape=(120,120,120), patch_size=self.patch_size_mri
            )
            mixed_age= self._reconstruct_from_patches_3d(
                mixed_age_patches, full_shape=(15,15,15), patch_size=self.patch_size_age
            ).unsqueeze(0)
            age1_vol = torch.full((15,15,15), age1, dtype=torch.float32)

        # Final output shape: img1, age1_vol, index, img2, age2_vol, different_j_index, mixed_img, mixed_age
        return (
            img1,
            age1_vol.unsqueeze(0),
            rid1,
            img2,
            age2_vol.unsqueeze(0),
            different_j_index,
            mixed_img.unsqueeze(0),
            mixed_age
        )

    def _split_into_patches_3d(self, volume, patch_size):
        """
        Split a 3D volume of shape (D,H,W) into patches of shape (patch_size, patch_size, patch_size).
        Returns: (num_patches, patch_size, patch_size, patch_size).
        For MRI => (120,120,120), patch_size=8 => 15x15x15 = 3375 patches
        For age => (15,15,15),   patch_size=1 => 15x15x15 = 3375 patches
        """
        D, H, W = volume.shape
        pd = D // patch_size
        ph = H // patch_size
        pw = W // patch_size

        patches = []
        for i in range(pd):
            for j in range(ph):
                for k in range(pw):
                    patch = volume[
                        i*patch_size:(i+1)*patch_size,
                        j*patch_size:(j+1)*patch_size,
                        k*patch_size:(k+1)*patch_size
                    ]
                    patches.append(patch)
        patches = torch.stack(patches, dim=0)  # (3375, patch_size, patch_size, patch_size)
        return patches

    def _random_mix_patches(self, img2_patches, img3_patches, age2_patches, age3_patches):
        """
        For each patch, randomly pick from img2 or img3 (and corresponding age).
        Shape => (3375, ...)
        """
        num_patches = img2_patches.shape[0]
        mixed_img_patches = []
        mixed_age_patches = []
        for i in range(num_patches):
            if random.random() < 0.5:
                mixed_img_patches.append(img2_patches[i])
                mixed_age_patches.append(age2_patches[i])
            else:
                mixed_img_patches.append(img3_patches[i])
                mixed_age_patches.append(age3_patches[i])

        mixed_img_patches = torch.stack(mixed_img_patches, dim=0)
        mixed_age_patches = torch.stack(mixed_age_patches, dim=0)
        return mixed_img_patches, mixed_age_patches

    def _reconstruct_from_patches_3d(self, patches, full_shape, patch_size):
        """
        Reassemble patches into shape (D,H,W).
        patches => (num_patches, patch_size, patch_size, patch_size)
        full_shape => (D, H, W)
        """
        D, H, W = full_shape
        pd = D // patch_size
        ph = H // patch_size
        pw = W // patch_size

        reconstructed = torch.zeros(D, H, W, dtype=patches.dtype)
        patch_idx = 0
        for i in range(pd):
            for j in range(ph):
                for k in range(pw):
                    reconstructed[
                        i*patch_size:(i+1)*patch_size,
                        j*patch_size:(j+1)*patch_size,
                        k*patch_size:(k+1)*patch_size
                    ] = patches[patch_idx]
                    patch_idx += 1

        return reconstructed

