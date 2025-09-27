import copy
import datetime
import os
import statistics
import sys
import time
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE

os.environ['CUDA_VISIBLE_DEVICES']='2,3'
from sklearn.metrics import confusion_matrix, accuracy_score
import wandb
import cv2
# import tensorflow as tf
import numpy as np
from utils import  MRIDataset_3D_three as MRIDataset
from utils import MRIDataset_inference

test_result=False
test_ad=False
local = True
if local:
    a = "./"
    if os.path.exists("../../../pub/liyifan/mri"):
        a="../../../pub/liyifan/mri"
    if os.path.exists("../../../home1/liyifan/mri"):#FS/
        a="../../../home1/liyifan/mri"
    if os.path.exists("../../../home2/liyifan/mri"):#FS/
        a="../../../home2/liyifan/mri"

else:
    a = '/storage/home/hcoda1/8/yli3863/scratch/'

# Define command-line arguments
parser = argparse.ArgumentParser(description='Model hyperparameters')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training (default: 32)')
parser.add_argument('--filters', type=int, nargs='+', default=[64, 128, 256, 512],
                    help='Number of filters for each convolutional layer (default: [64, 128, 256, 512])')
parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for convolutional layers (default: 3)')
parser.add_argument('--activation', type=str, default='silu', choices=['relu', 'sigmoid', 'tanh', 'silu'],
                    help='Activation function for convolutional layers (default: silu)')
parser.add_argument('--age_activation', type=str, default='silu', choices=['relu', 'sigmoid', 'tanh', 'silu'],
                    help='Activation function for age prediction (default: silu)')
parser.add_argument('--structure_vec_size', type=int, default=256, help='Size of the structure vector (default: 100)')
parser.add_argument('--longitudinal_vec_size', type=int, default=256,
                    help='Size of the longitudinal vector (default: 100)')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for temperature scaling (default: 0.7)')
parser.add_argument('--restore_from_checkpoint', action='store_true',
                    help='Whether to restore model from checkpoint (default: False)')
parser.add_argument('--prefetch_buffer_size', type=int, default=3, help='Size of the prefetnch buffer (default: 3)')
parser.add_argument('--shuffle_buffer_size', type=int, default=1000, help='Size of the shuffle buffer (default: 1000)')
parser.add_argument('--input_height', type=int, default=256, help='Height of input images (default: 256)')
parser.add_argument('--input_width', type=int, default=256, help='Width of input images (default: 256)')
parser.add_argument('--input_channel', type=int, default=1, help='Number of input channels (default: 1)')
parser.add_argument('--structure_vec_similarity_loss_mult', type=int, default=1,
                    help='Multiplier for structure vector similarity loss (default: 100)')
parser.add_argument('--age_loss_mult', type=int, default=1, help='Multiplier for age prediction loss (default: 100)')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs (default: 5000)')
parser.add_argument('--checkpoint_save_interval', type=int, default=5,
                    help='Interval for saving checkpoints (default: 5)')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 1e-4)')
parser.add_argument('--experiment_name', type=str, default='proposed_method',
                    help='Name of the experiment (default: proposed_method)')
parser.add_argument('--seed', type=int, default=1234, help='Seed (default: 1234)')
parser.add_argument('--cuda', type=str, default='0', help="which gpu")
parser.add_argument('--resize_dim', type=int, default=128, help="the resize shape")
parser.add_argument('--save_dir', type=str, default=a + '/FS/model_save/',
                    help="the directory to save models and checkpoints")

# Parse command-line arguments
args = parser.parse_args()
stage0_epoch=0
stage1_epoch=300
vae=False
c_margin=1
devices= args.cuda
# Assign parsed arguments to variables
BATCH_SIZE = args.batch_size
FILTERS = args.filters
KERNEL_SIZE = args.kernel_size
ACTIVATION = getattr(F, args.activation)
AGE_ACTIVATION = getattr(F, args.age_activation)
LAST_ACTIVATION = torch.sigmoid
STRUCTURE_VEC_SIZE = args.structure_vec_size
LONGITUDINAL_VEC_SIZE = args.longitudinal_vec_size
TEMPERATURE = args.temperature
RESTORE_FROM_CHECKPOINT = args.restore_from_checkpoint
PREFETCH_BUFFER_SIZE = args.prefetch_buffer_size
SHUFFLE_BUFFER_SIZE = args.shuffle_buffer_size
INPUT_HEIGHT = args.input_height
INPUT_WIDTH = args.input_width
INPUT_CHANNEL = args.input_channel
STRUCTURE_VEC_SIMILARITY_LOSS_MULT = args.structure_vec_similarity_loss_mult
AGE_LOSS_MULT = args.age_loss_mult
EPOCHS = args.epochs
CHECKPOINT_SAVE_INTERVAL = args.checkpoint_save_interval
LR = args.lr
EXPERIMENT_NAME = args.experiment_name
resize_dim = args.resize_dim
save_dir = args.save_dir

# Set memory growth to true
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from torchvision import models
reserved_memories = []
def occupy_gpu_memory(reserve_ratio=0.95):
    """
    在 `CUDA_VISIBLE_DEVICES` 指定的 GPU 上创建大张量占满显存，但仍然允许 PyTorch 复用显存。
    `reserve_ratio` 控制预留多少比例的显存（默认 90%）。
    """
    global reserved_memories

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清除缓存

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible_devices:
            print("No GPUs are set in CUDA_VISIBLE_DEVICES.")
            return

        gpu_indices = list(map(int, visible_devices.split(",")))  # 解析成列表，例如 [2,3]
        current_device = torch.cuda.current_device()  # 记录当前默认 GPU

        for i, gpu_id in enumerate(gpu_indices):  # 遍历 CUDA_VISIBLE_DEVICES 里指定的 GPU
            torch.cuda.set_device(i)  # 选择 PyTorch 看到的 `cuda:i`，实际对应物理 GPU `gpu_id`
            total_memory = torch.cuda.get_device_properties(i).total_memory  # 获取该 GPU 的总显存
            reserved_memory = int(total_memory * reserve_ratio)  # 预留 `reserve_ratio` 这么多显存

            try:
                # **创建一个大张量，但 `requires_grad=False` 允许 PyTorch 复用**
                tensor = torch.ones((reserved_memory // 4,), dtype=torch.float32, device=f"cuda:{i}", requires_grad=False)
                reserved_memories.append(tensor)

                print(f"Reserved {reserved_memory / 1024**3:.2f} GB on cuda:{i} (Physical GPU {gpu_id}).")

            except RuntimeError:
                print(f"Warning: Could not reserve full memory on cuda:{i} (Physical GPU {gpu_id}), possibly out of memory.")

        torch.cuda.set_device(current_device)  # 恢复原来的默认设备

        print("GPU memory successfully reserved and is available for reuse.")
    del reserved_memories


class PerceptualLoss(nn.Module):
    def __init__(self,
                 stage1=True,
                 pretrained_path="./resnet_10.pth"):
        """
        stage1=True 时，启用3D感知损失 (MedicalNet)。
        pretrained_path: MedicalNet的3DResNet10预训练权重路径.
        """
        super(PerceptualLoss, self).__init__()
        self.stage1 = stage1

        if stage1 is not None:
            from MedicalNet.models.resnet import resnet10
            #
            # # 1) 构建3D ResNet10, 让 in_channels=1 以适配MRI单通道
            # #    如果你的权重是3通道，可以先把in_channels=3，然后strict=False加载
            # net_3d =nn.DataParallel(resnet10(
            #     sample_input_D=120,  # 你的MRI深度
            #     sample_input_H=120,
            #     sample_input_W=120,
            #     shortcut_type='B',
            #     no_cuda=False,
            #     num_seg_classes=1))
            #
            # # 2) 加载预训练权重 (strict=False，以防一些层对不上)
            # ckpt = torch.load(pretrained_path)
            # if "state_dict" in ckpt:
            #     ckpt = ckpt["state_dict"]
            # missing, unexpected = net_3d.load_state_dict(ckpt, strict=False)
            # print("Missing keys:", missing)
            # print("Unexpected keys:", unexpected)
            # base=net_3d.module
            # # 3) 设为 eval + 不参与梯度
            # self.net_3d = nn.Sequential(
            #     base.conv1,
            #     base.bn1,
            #     base.relu,
            #     base.maxpool,  # 可选：通常保留maxpool以匹配后续layer的输入形状
            #     base.layer1,
            #     base.layer2
            # )
            # self.net_3d.cuda().eval()
            # for param in self.net_3d.parameters():
            #     param.requires_grad = False

        else:
            self.net_3d = None  # stage1=False时，不使用感知网络

    def forward(self,
                target_img,
                input_img,
                model,
                sturcture_n1,
                state_n1,
                age,
                state="train",
                stage1=None):
        """
        参数说明：
         - target_img, input_img:  形状 [B,1,120,120,120]
         - model, sturcture1, state_n1, age: 只在 else 分支用
         - state, stage1: 训练过程中的状态控制
        """
        # 如果在 forward 的时候传入 stage1，可覆盖
        if stage1 is not None:
            self.stage1 = stage1

        if self.stage1 is None:
            # 不计算损失
            return torch.tensor(0.0, device=input_img.device)

        if self.stage1:
            # =========== Stage1: 用 3D ResNet10 计算感知特征 + 重构损失 =============

            # (可选) 如果权重的第一层是3通道，而你不想改网络结构，
            # 可以把单通道MRI -> 3通道:
            # input_img  = input_img.repeat(1,3,1,1,1)
            # target_img = target_img.repeat(1,3,1,1,1)

            # # with torch.no_grad():
            # feat_in = self.net_3d(input_img)  # [B, C', D', H', W']
            # feat_tar = self.net_3d(target_img)
            # loss_perc = F.mse_loss(feat_in, feat_tar, reduction="mean")

            # 再做一个重构损失(像你原来的 rec_loss)
            rec_loss = F.mse_loss(input_img, target_img, reduction="mean")

            return  rec_loss * 500#+loss_perc

        else:
            # =========== Stage2: 你原先的 code，做 fine-tune decoder =============
            structure1, state1, age_hat1 = model(target_img)
            loss_state = F.mse_loss(state_n1,state1, reduction="mean")+ F.mse_loss(structure1[-2],sturcture_n1[-2], reduction="mean")

            loss_age = F.mse_loss(age, age_hat1)
            loss = loss_age
            rec_loss = F.mse_loss(input_img, target_img, reduction="mean")
            wandb.log({
                state + "/state_diff": loss_state,
                # state + "/str_diff": loss_str,
                state + "/rec_mse": rec_loss,
                state + "/rec_loss_age": loss_age
            })
            # with torch.no_grad():
            # feat_in = self.net_3d(input_img)  # [B, C', D', H', W']
            # feat_tar = self.net_3d(target_img)
            #loss_perc = F.mse_loss(feat_in, feat_tar, reduction="mean")

            # 再做一个重构损失(像你原来的 rec_loss)

            return loss/100 + rec_loss * 500  + loss_state#+loss_perc
def log_print(dir, msg, add_timestamp=False):
    if not isinstance(msg, str):
        msg = str(msg)
    if add_timestamp:
        msg += " (logged at {})".format(
            datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    with open(os.path.join(dir, "logs.txt"), "a+") as log_file:
        log_file.write(msg + "\n")

erase=transforms.Compose([
    transforms.RandomErasing(p=0.5, scale=(0.05, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)
            ])
#First round unchanged, after learning the first round, predict all rid and j features
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
def visualize_tsne(feature, clinical_age1_, epoch, save_path='./',str=False):
    """
    Visualize features using t-SNE and encode colors based on clinical age.

    Parameters:
    - feature: n*256 numpy array representing feature data
    - clinical_age1_: numpy array of length n representing corresponding clinical ages
    - epoch: integer, current training epoch number for file naming
    - save_path: string, path to save the image, default is current directory

    Returns:
    - Path of the saved image
    """

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    if len(feature.shape)>2:
        feature = feature.reshape(feature.shape[0], -1)
    feature_tsne = tsne.fit_transform(feature)

    if str:
        # 根据不同临床年龄生成唯一颜色和形状
        plt.figure(figsize=(10, 8))
        unique_ages = np.unique(clinical_age1_)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', '*', 'X', 'h']  # 不同形状
        num_shapes = len(markers)
        colors = [plt.get_cmap('tab10')(i / len(unique_ages)) for i in range(len(unique_ages))]

        for i, age in enumerate(unique_ages):
            age_mask = np.array(clinical_age1_) == age
            plt.scatter(
                feature_tsne[age_mask, 0],
                feature_tsne[age_mask, 1],
                c=[colors[i]],  # 直接使用 colors 列表中的颜色
                label=f'id {age}',
                marker=markers[i % num_shapes],
                s=100  # 设置点的大小
            )

        plt.legend()  # 显示图例
        file_name = f'epoch_{epoch}_str_train.png'
    else:
        file_name = f'epoch_{epoch}_age_train.png'

    # 将 clinical_age1_ 标准化为 0 到 1 之间的值，用于颜色映射
        norm = plt.Normalize(clinical_age1_.min(), clinical_age1_.max())

        # 创建图形并绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=clinical_age1_, cmap='viridis', norm=norm)


        # 添加颜色条
        plt.colorbar(scatter, label='Clinical Age')

        plt.title(f"t-SNE Visualization of Features with Clinical Age Color Coding - Epoch {epoch}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

        # 构建文件名并保存图片

    file_path = save_path +'/'+ file_name
    plt.savefig(file_path)

    # 显示并关闭图形
    # plt.show()
    plt.close()

    return file_path
def visualize_age_scatter(age_, clinical_age1_, epoch, save_path='./'):
    """
    画出 age_ 和 clinical_age1_ 的散点图，并将其保存为图片。

    参数:
    - age_: 横坐标数据 (numpy array)
    - clinical_age1_: 纵坐标数据 (numpy array)
    - epoch: 当前训练的 epoch 号，用于文件命名
    - save_path: 保存图片的路径, 默认为当前目录

    返回:
    - 保存的图片路径
    """

    # 创建图形并绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(age_, clinical_age1_, c='b', marker='o', alpha=0.5)
    plt.xlabel('Age (Predicted)')
    plt.ylabel('Absolute Age')
    plt.title(f'Scatter plot of Age vs Absolute Age (Epoch {epoch})')

    # 保存图片
    save_file = f'{save_path}/epoch_{epoch}_age_scatter.png'
    plt.savefig(save_file)
    plt.close()

    # print(f"Scatter plot saved at {save_file}")


def visualize_age_matrix_scatter(matrix1, matrix2, epoch, save_path='./',head="original_gap"):
    """
    画出两个 batch x 15 x 15 x 15 矩阵的 [:, 7, :, :] 切片中对应位置的 batch 对数据的散点图，
    并将其保存为图片。

    参数:
    - matrix1: 第一个 batch x 15 x 15 x 15 的矩阵 (numpy array)
    - matrix2: 第二个 batch x 15 x 15 x 15 的矩阵 (numpy array)
    - epoch: 当前训练的 epoch 号，用于文件命名
    - save_path: 保存图片的路径, 默认为当前目录

    返回:
    - 保存的图片路径
    """
    # 创建保存路径（如果不存在）
    # os.makedirs(save_path, exist_ok=True)

    # 提取 [:, 7, :, :] 切片
    slice1 = matrix1[:, 7, :, :].cpu().detach().numpy()
    slice2 = matrix2[:, 7, :, :].cpu().detach().numpy()

    # 创建图形
    fig, axes = plt.subplots(15, 15, figsize=(20, 20))

    # 绘制散点图
    for i in range(15):
        for j in range(15):
            ax = axes[i, j]
            ax.scatter(slice1[:, i, j], slice2[:, i, j], c='b', alpha=0.5, s=5)
            ax.set_xticks([])
            ax.set_yticks([])

    # 设置整体标题
    plt.suptitle(head+f'(_Epoch {epoch})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图片
    save_file = os.path.join(save_path, head+f'_epoch_{epoch}.png')
    plt.savefig(save_file)
    plt.close()

    return save_file


def Pretext(model_encode,model_decode, optimizer, Epochs, train_dataset, val_loader, test_loader, structure_loss_mult, age_loss_mult,
            save_dir,percep=None, mci_loader=None,ad_loader=None):

    os.mkdir(save_dir + "/image/")
    step = 0

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)
    total_loss_, image_similarity_loss_, structure_vec_sim_loss_, age_prediction_loss_ = [], [], [], []
    train_in_dataset=MRIDataset_inference(train_dir,resize_dim)
    train_inf_loader = torch.utils.data.DataLoader(train_in_dataset,batch_size=args.batch_size, shuffle=False, num_workers=32)
    semi_result = []
    percep = PerceptualLoss(stage1=True)
    for epoch in range(Epochs):
        if epoch<stage0_epoch:
            model_encode.train()
            model_decode.train()
            print("stage 0 initialize")
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True)
        elif epoch >=stage0_epoch and epoch<stage1_epoch+stage0_epoch:
            print("stage 1 training")
            percep = PerceptualLoss(stage1=False)
            model_encode.eval()
            model_decode.train()
            if epoch ==stage0_epoch:
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=int(args.batch_size / 2), shuffle=True,
                                                           num_workers=32, pin_memory=True)
        total_loss, image_similarity_loss, structure_vec_sim_loss, age_prediction_loss = [], [], [], []

        model_save_path = os.path.join(save_dir,
                                       f'check_point_model_{epoch}_{structure_loss_mult}_{age_loss_mult}.pth')

        for batch_id, (img1, age1, index , img2, age2, different_j_index,mixed_img,mixed_age) in enumerate(tqdm(train_loader)):
            if epoch % 10 == 0 and batch_id==0:
                from ae_3d import UNetModel3D
                model=UNetModel3D()
                model.decoder=model_decode.cpu()
                model.encoder=model_encode.cpu()
                torch.save(model.state_dict(), model_save_path)
                # if epoch % 10 == 0:
                #     torch.save(optimizer.state_dict(), optimizer_save_path)
                model_decode.cuda()
                model_encode.cuda()
                evluate(model_encode,model_decode, val_loader,epoch,percep)
                # Save the hidden feature
                # hidden_save_path = os.path.join(save_dir, f'hidden_feature_{epoch}_{structure_loss_mult}_{age_loss_mult}.pth')
                # if epoch % 10 == 0:
                #     torch.save({'y1': y1, "structure1": structure1, 'state1': state1, 'age_hat1': age_hat1, 'y2': y2,
                #                 "structure2": structure2, 'state2': state2, 'age_hat2': age_hat2}, hidden_save_path)

                test(model_encode,model_decode, test_loader, save_dir,epoch=epoch)
                model_encode.train()
                model_decode.train()

            X1_input = img1.cuda()
            # print(torch.max(X1))
            X2_input = img2.cuda()
            y1 = age1[:,0,0,0].cuda()
            y2 = age2[:,0,0,0].cuda()
            # mixed_img =mixed_img.cuda()
            # mixed_age = mixed_age.cuda()
            age_diff = (y1.float() - y2.float()) # 添加一个维度，形状变为 (batch_size, 1)

            # 重复 256 次
            structure1, state1,  age_hat1 = model_encode(X1_input)
            structure2, state2,  age_hat2 = model_encode(X2_input)
            # structure_mixed, state_mixed, mixed_age_hat= model_encode( mixed_img)
            predicted_img1 = model_decode(structure2,state2 , -age_hat2[:,0]+ age_hat1[:,0])
            # predicted_img2= model_decode(structure1, state1, age_hat2[:,0,::]- age_hat1[:,0,::])

            batch_size, C,Z, H, W = X1_input.size()

            trip_c =0
            loss_age_gap = F.mse_loss(y1.float() - y2.float(), -age_hat2[:, 0] + age_hat1[:, 0])
            loss_age_mean = ((age_hat1[:, 0] + age_hat2[:, 0] - y1 - y2).sum() / (3 * y1.shape[0])) ** 2 / 10
            loss_age = loss_age_gap + loss_age_mean
            recon_loss = percep(predicted_img1, X1_input, model_encode,structure1, state1, age_hat1)
            # recon_loss2=percep(predicted_img_mix, mixed_img, model_encode,structure_mixed, state_mixed, mixed_age_hat)
            # img_diff_loss=F.mse_loss(predicted_img1-predicted_img2,X1_input-X2_input,reduction="mean")*50
            loss_rec = recon_loss #+ recon_loss2
            if epoch<stage1_epoch+stage0_epoch:
                loss = loss_rec+  loss_age
                wandb.log({"train_stage0/loss_age_gap": loss_age_gap, "train_stage0/loss_age_mean": loss_age_mean,
                           "train_stage0/loss_rec": loss_rec
                           })
            else:
                loss = loss_rec
            # wandb.log({"train/loss_age": loss_age, "train/trip_c":trip_c,  "train/trip_time":trip_time,
            #           "train/rec_loss_age": kld_loss,"train/str_distance_positive":str_distance_positive})
            if epoch % 10 == 0 and batch_id==0:
                # Save the model's state_dict (recommended way in PyTorch)
                title=[]
                for i in range(len(age1)):
                    title.append(str(age1[i][0,0].cpu().numpy())[0:5] + '_id:' + str(index[i]))
                title1=[]
                for i in range(len(mixed_age)):
                    title1.append(str(mixed_age[i][0,0].cpu().numpy())[0:5] + '_id:' + str(different_j_index[i]))
                generate_images(predicted_img1, title, f"{epoch}_train_decode", save_dir + "/image/")
                generate_images(X1_input, title, f"{epoch}_train_input", save_dir + "/image/")
                # generate_images(predicted_img_mix, title1, f"{epoch}_train_decode_mix", save_dir + "/image/")
                # generate_images(mixed_img, title1, f"{epoch}_train_input_mix", save_dir + "/image/")
                # visualize_age_matrix_scatter(y1.float() - y2.float(),  -age_hat2[:,0,::]+ age_hat1[:,0,::], save_path=save_dir + "/image/",epoch=epoch,head="orginal_gap")
                # # visualize_age_matrix_scatter(y1.float() - mixed_age.float(),  -mixed_age_hat[:,0,::]+ age_hat1[:,0,::], save_path=save_dir + "/image/",epoch=epoch,head="mixed_gap")
                # visualize_age_matrix_scatter(y1.float() ,
                #                              age_hat1[:,0,::], save_path=save_dir + "/image/",epoch=epoch,
                #                              head="absolute_age")

                # visualize_tsne(torch.concat([state1,state2],0).cpu().detach().numpy(), torch.concat([age1,age2],0), epoch, save_path=save_dir)
                # visualize_tsne(torch.concat([structure1[-2],structure2[-2]],0).cpu().detach().numpy(),index+index, epoch, save_path=save_dir, str=True)
                # visualize_age_scatter(age1, clinical_age1, epoch, save_path=save_dir)
                visualize_age_scatter( age_hat1[:, 0].cpu().detach().numpy(),y1.cpu().detach().numpy(),
                                      str(epoch) + 'predicted_aging_index_vs_actual', save_path=save_dir)
            loss_structure = trip_c
            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_encode.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(model_decode.parameters(), max_norm=0.5)
            optimizer.step()
            # total_loss.append(loss.item())
            # structure_vec_sim_loss.append(trip_c)
            # age_prediction_loss.append(loss_age.item())


        # print(y1 - age_hat1[:, 0])

        # total_loss_.append(total_loss)
        # structure_vec_sim_loss_.append(structure_vec_sim_loss)
        # age_prediction_loss_.append(age_prediction_loss)

        print(
            f"Epoch_{epoch}: loss_age: {loss_age}  loss_structure: {loss_structure}, loss_rec:{loss_rec},  loss: {loss}")


    return model, total_loss, 0, structure_vec_sim_loss, age_prediction_loss, semi_result


def evluate(model_encode,model_decode, val_loader,epoch=0,percep=None):
    # freeze
    model_encode.eval()
    model_decode.eval()
    loss_age, loss_structure = [], []

    with torch.no_grad():
        for (img1, age1, index, img2, age2, different_j_index,mixed_img,mixed_age) in val_loader:
            X1 = img1.cuda()
            X2 = img2.cuda()
            y1=age1[:,0,0,0]
            y2=age2[:,0,0,0]
            structure1,state_n1, age_hat1= model_encode(X1)
            structure2, state_n2 , age_hat2= model_encode(X2)

            age_diff = (y1.float() - y2.float())  # 添加一个维度，形状变为 (batch_size, 1)
            # 重复 256 次
            loss_age.append( ((torch.norm(y1.cuda() - age_hat1[:,0]) ** 2 )/ ( y1.shape[0])).cpu().detach().numpy())

            # structure1 = structure1.cpu()
            # structure2 = structure2.cpu()

            predicted_img1=model_decode(structure2,state_n2 ,age_diff.cuda())

            loss_structure.append(torch.mean(((structure1[-2]) - (structure2[-2])) ** 2).cpu().detach().numpy())

            rec = percep(predicted_img1, X1,model_encode,structure1,state_n1,age_hat1,state="val")

    wandb.log({"val/loss_age": np.mean(loss_age), "val/loss_structure": np.mean(loss_structure), "val/loss_rec": rec})
    generate_images(predicted_img1, y1, f"{epoch}_test_decode", save_dir + "/image/")
    generate_images(X1, y1, f"{epoch}_test_input", save_dir + "/image/")
    # model.train()
    torch.cuda.empty_cache()
    return None


def test(model_encode,model_decode, val_loader, save_dir,epoch=0):
    # freeze
    model_encode.eval(),model_decode.eval()

    loss_age, loss_structure = [], []
    ages = {'y': [], 'age_hat': []}
    ab_age_loss=[]
    with torch.no_grad():
        for (img1, y1, index, img2, y2, different_j_index,mixed_img,mixed_age)  in val_loader:
            X1 = img1.cuda()
            X2 = img2.cuda()
            y1=y1[:,0,0,0]
            y2=y2[:,0,0,0]
            structure1,state1 ,age_hat1,= model_encode(X1)
            structure2,state2, age_hat2= model_encode(X2)
            loss_age.append(((torch.norm(y1.cuda() - age_hat1[:,0] - y2.cuda() + age_hat2[:,0]) ** 2) / (y1.shape[0])).cpu().detach().numpy())
            ab_age_loss.append(F.l1_loss(y1.cuda(),  age_hat1[:,0], reduction='mean').cpu().detach().numpy())

            # structure1 = structure1.numpy()
            # # structure1 = structure1.numpy()
            age_diff = (y1.float() - y2.float())  # 添加一个维度，形状变为 (batch_size, 1)
            age_hat1 = age_hat1.cpu().numpy().flatten()
            age_hat2 = age_hat2.cpu().numpy().flatten()
            y1 = y1.numpy().flatten()
            y2 = y2.numpy().flatten()
            ages['y'].extend(y1)
            ages['y'].extend(y2)
            ages['age_hat'].extend(age_hat1)
            ages['age_hat'].extend(age_hat2)


            # 重复 256 次
            predicted_img1=model_decode(structure2,state2, age_diff.cuda())

            # structure1 = structure1.cpu()
            # structure2 = structure2.cpu()

            loss_structure.append(torch.mean(((structure1 [-2])- (structure2) [-2])** 2).cpu().detach().numpy())
    # model.train()

    rec= F.mse_loss(
        predicted_img1  , X1 , reduction="mean")
    log_print(save_dir, f"Test age: {np.mean(loss_age)}, structure: {np.mean(loss_structure)}, rec_loss: {rec}")

    generate_images(predicted_img1, y1, f"{epoch}_test_decode", save_dir + "/image/")
    generate_images(X1, y1, f"{epoch}_test_input", save_dir + "/image/")
    print(f"Test: {np.mean(loss_age)}")
    print(f"Test absolute: {np.mean(ab_age_loss)}")

    wandb.log({"test/loss_age": np.mean(loss_age), "test/loss_structure": np.mean(loss_structure), "test/loss_rec": rec })
    torch.cuda.empty_cache()
    return None


def generate_images(predicted_imgs, age1, image_name, path):
    # 将 predicted_imgs 转为 numpy 数组
    predicted_imgs = predicted_imgs.cpu().detach().numpy()
    batch_size = predicted_imgs.shape[0]
    img_height, img_width = resize_dim,resize_dim

    # 初始化用于保存水平拼接后的图片的列表
    hseq = []

    for i in range(batch_size):
        vseq = []

            # 获取图片并进行处理
        img = predicted_imgs[i,0, :, :, 60]  # 选择第一个通道，假设图片是单通道
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 将图片转换为3通道以便显示文本

        # 在图片下方显示年龄
        age_text = f"Age: {age1[i]}"
        cv2.putText(img, age_text, (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.2, (255, 255, 255), 1, cv2.LINE_AA)

        vseq.append(img)

        # 垂直拼接图片
        vseq = np.vstack(vseq)
        hseq.append(vseq)

    # 水平拼接
    hseq = np.hstack(hseq)

    # 创建保存路径并保存图片
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, f"{image_name}.png")
    cv2.imwrite(save_path, hseq)


if __name__ == "__main__":
    occupy_gpu_memory()

    if local:
        a_ = "./"
        if os.path.exists("../../../pub/liyifan/mri"):
            a_ = "../../../pub/liyifan/mri"
        elif os.path.exists("../../../home1/liyifan/mri"):
            a_ = "../../../home1/liyifan/mri"
    else:
        a_ = '/storage/home/hcoda1/8/yli3863/scratch/'

    if test_result==True:
        save_path = a_ + "/tests"
    else:
        save_path = a_ + "/3dunet"

    save_path_ = os.path.join(save_path, "ex_{}".format(
        datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    ))
    if test_result==True:
        temp="_test"
    else:
        temp="_vae"

    wandb.init(
        # set the wandb project where this run will be logged
        project="3dUnet_255_all_flip"+temp,

        # track hyperparameters and run metadata
        config={
            "save_path_ ": save_path_ ,
            "architecture": "train with val",
            "resume":"/pub/liyifan/mri/3dunet/ex_2025_02_26_08-43-39/check_point_model_290_1_1.pth"

        }
    )
    path_model = "/pub/liyifan/mri/3dunet/ex_2025_02_26_08-43-39/check_point_model_290_1_1.pth"
    state_dict = torch.load(path_model)
    new_state_dict = {k.replace('.module.', '.'): v for k, v in state_dict.items()}

    from ae_3d import UNetDecoder3D, UNetEncoder3D


    image_size = 128
    model_encode = nn.DataParallel(UNetEncoder3D())
    model_decode = nn.DataParallel(UNetDecoder3D())
    encoder_state_dict = {k.replace("encoder.", ""): v for k, v in new_state_dict.items() if k.startswith("encoder.")}
    decoder_state_dict = {k.replace("decoder.", ""): v for k, v in new_state_dict.items() if k.startswith("decoder.")}

    # 加载权重
    model_encode.module.load_state_dict(encoder_state_dict)
    model_decode.module.load_state_dict(decoder_state_dict)
    if not os.path.exists(save_path_):
        if not (os.path.exists(os.path.join(save_path))):
            os.mkdir(os.path.join(save_path))
        os.mkdir(os.path.join(save_path_))
    source_code_file = os.path.abspath(__file__)
    import shutil

    save_dir = save_path_
    shutil.copy2(source_code_file, save_path_)
    shutil.copy2("ae_3d.py", save_path_)
    shutil.copy2("utils.py", save_path_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # dataset
    train_dir = a + '/FS_2/processed_health_all/train_val/'
    test_dir = a + '/FS_2/processed_health_all/test/'
    val_dir = a + '/FS_2/processed_health_all/validation/'
    train_dir_mci = a + '/FS_2/processed_MCI_all/train/'
    test_dir_mci = a + '/FS_2/processed_MCI_all/test/'
    val_dir_mci = a + '/FS_2/processed_MCI_all/validation/'
    train_dir_ad= a + '/FS_2/processed_AD_all/train/'
    test_dir_ad= a + '/FS_2/processed_AD_all/test/'
    val_dir_ad = a + '/FS_2/processed_AD_all/validation/'

    train_mci_ad=a+ '/FS_2/processed_mc_ad_all/train/'
    train_h_ad= a + '/FS_2/processed_h_ad_all/test/'
    train_dataset= MRIDataset( train_dir, resize_dim,train=True)
    train_inference = MRIDataset_inference(train_dir, resize_dim)


    val_loader = torch.utils.data.DataLoader(MRIDataset( val_dir, resize_dim,train=False),
                                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(MRIDataset( test_dir, resize_dim,train=False,mci_ad=True),
                                              batch_size=args.batch_size, shuffle=False, num_workers=0)
    mci_loader=torch.utils.data.DataLoader(MRIDataset( train_dir_mci, resize_dim),
                                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    ad_loader=torch.utils.data.DataLoader(MRIDataset( train_dir_ad, resize_dim),
                                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    mci_ad_loader=torch.utils.data.DataLoader(MRIDataset( train_mci_ad, resize_dim,mci_ad=False),
                                               batch_size=args.batch_size, shuffle=True, num_workers=0)
    h_ad_loader=torch.utils.data.DataLoader(MRIDataset( train_h_ad, resize_dim,mci_ad=False),
                                              batch_size=args.batch_size, shuffle=True, num_workers=0)

    if test_result:
        from ae_3d import UNetModel3D
        model=nn.DataParallel(UNetModel3D())
        path_model ="/pub/liyifan/mri/3dunet/ex_2025_02_26_08-43-39/check_point_model_290_1_1.pth"
        state_dict = torch.load(path_model)
        new_state_dict = {k.replace('.module.', '.'): v for k, v in state_dict.items()}
        model.module.load_state_dict(new_state_dict)
        model.cuda()
        batchsize =8
        if test_ad==True:
            from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

            # 假设你已经有: model, test_loader(负样本), ad_loader(正样本)

            # 存储所有的 MSE 和对应标签
            mse_scores_neg = []
            mse_scores_pos = []

            model.eval()

            # ============ 1. 处理负样本 (类别=0) ============
            for (img1, age1, index, img2, age2, different_j_index, mixed_img, mixed_age) in test_loader:
                y1 = age1[:, 0, 0, 0]
                y2 = age2[:, 0, 0, 0]

                # 过滤一些不满足条件的 index
                valid_indices = (np.array(y2) > np.array(y1)) & (np.array(y2) < np.array(y1) + 999)
                img1 = img1[valid_indices]
                img2 = img2[valid_indices]
                y1 = y1[valid_indices]
                y2 = y2[valid_indices]

                if len(y1) == 0:
                    continue

                with torch.no_grad():
                    # (1) 编码
                    structure1, state1, age_hat1 = model.module.encoder(img1.cuda())
                    # (2) 计算真实年龄差, 并解码
                    age_diff = (y2.float() - y1.float()).cuda()
                    predicted_img2 = model.module.decoder(structure1, state1, age_diff)
                    # (3) 计算 MSE
                    mse_vals = F.mse_loss(
                        img2.cuda(),
                        predicted_img2,
                        reduction="none"
                    ).mean(dim=(1, 2, 3, 4)).cpu().numpy()

                mse_scores_neg.append(mse_vals)

            # ============ 2. 处理正样本 (类别=1) ============
            for (img1, age1, index, img2, age2, different_j_index, mixed_img, mixed_age) in h_ad_loader:
                y1 = age1[:, 0, 0, 0]
                y2 = age2[:, 0, 0, 0]

                valid_indices = (np.array(y2) > np.array(y1)) & (np.array(y2) < np.array(y1) + 999)
                img1 = img1[valid_indices]
                img2 = img2[valid_indices]
                y1 = y1[valid_indices]
                y2 = y2[valid_indices]

                if len(y1) == 0:
                    continue

                with torch.no_grad():
                    structure1, state1, age_hat1 = model.module.encoder(img1.cuda())
                    age_diff = (y2.float() - y1.float()).cuda()
                    predicted_img2 = model.module.decoder(structure1, state1, age_diff)
                    mse_vals = F.mse_loss(
                        img2.cuda(),
                        predicted_img2,
                        reduction="none"
                    ).mean(dim=(1, 2, 3, 4)).cpu().numpy()

                mse_scores_pos.append(mse_vals)

            # 合并到一个数组里
            mse_scores_neg = np.concatenate(mse_scores_neg, axis=0) if len(mse_scores_neg) > 0 else np.array([])
            mse_scores_pos = np.concatenate(mse_scores_pos, axis=0) if len(mse_scores_pos) > 0 else np.array([])

            all_scores = np.concatenate([mse_scores_neg, mse_scores_pos], axis=0)
            all_labels = np.array([0] * len(mse_scores_neg) + [1] * len(mse_scores_pos))

            print(f"Total negative samples: {len(mse_scores_neg)}")
            print(f"Total positive samples: {len(mse_scores_pos)}")

            # ============ 3. 计算 ROC AUC ============
            auc_val = roc_auc_score(all_labels, all_scores)
            print("ROC AUC:", auc_val)

            # ============ 4. 计算并可视化 ROC 曲线 (可选) ============
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
            # 你可以自行画图，或根据需要寻找最佳阈值
            # 例如用 Youden’s J statistic: max(tpr - fpr)
            j_scores = tpr - fpr
            best_threshold_idx = np.argmax(j_scores)
            best_thresh = thresholds[best_threshold_idx]
            print(f"Best threshold by Youden’s J: {best_thresh:.4f}")

            # ============ 5. 计算准确率、错误率等 (基于某个阈值) ============
            # 假设我们用 best_thresh 进行二分类
            preds = (all_scores >= best_thresh).astype(int)
            acc = accuracy_score(all_labels, preds)
            err = 1 - acc

            print(f"Accuracy at threshold={best_thresh:.4f}: {acc:.4f}")
            print(f"Error rate at threshold={best_thresh:.4f}: {err:.4f}")
        k = 0
        age_loss_with_diff = 0
        age_loss_direct = 0
        age_loss_base = 0
        age_loss_base_absolute = 0
        rec_loss_with_diff = 0
        rec_loss_direct = 0
        age_loss_direct_regressed = 0
        rec_loss_base = 0
        # model.module.set_trainable_downsample_conv(False)
        # model.module.set_trainable_upample_conv(False)
        model.eval()
        idx = 0
        age1_=[]
        age2_=[]
        mixed_age_=[]
        age1_hat_=[]
        age2_hat_=[]
        mixed_age_hat_=[]
        structure_list=[]
        idx_list=[]
        age_hat_list=[]
        extra = True
        extra_3d=True
        structure_plot=False
        fastsufer=False
        mse_list_ = []
        if fastsufer:

            FASTSURFER_DIR = "./fastsurfer/my_mri_data/"
            GEN_DIR = os.path.join(FASTSURFER_DIR, "generated")
            ORIG_DIR = os.path.join(FASTSURFER_DIR, "original")
            OLD_DIR=os.path.join(FASTSURFER_DIR,"old")
            import nibabel as nib
            # 确保目录存在
            os.makedirs(GEN_DIR, exist_ok=True)
            os.makedirs(ORIG_DIR, exist_ok=True)
            os.makedirs(OLD_DIR, exist_ok=True)


        SUBJECT_LIST_PATH = "./fastsurfer/subject_list.txt"
        CLEAN_SUBJECT_LIST_PATH = "./fastsurfer/clean_subject_list.txt"
        with open(SUBJECT_LIST_PATH, "w") as subject_list, open(CLEAN_SUBJECT_LIST_PATH,
                                                                "w") as clean_subject_list:
            for (img1, age1, index , img2, age2, different_j_index,mixed_img,mixed_age) in test_loader:
                y2=age2[:,0,0,0]
                y1=age1[:,0,0,0]
                age2=y2
                age1=y1
                rid=np.array(list(index))
                valid_indices = (np.array(y2) > np.array(y1)+0)&  (np.array(y2) < np.array(y1)+999)

                # 如果没有满足条件的索引，跳过该 batch
                if not valid_indices.any():
                    continue
                idx = idx + 1
                # indices = valid_indices.nonzero(as_tuple=False).squeeze().tolist()
                index = [valid_indices[i] for i in valid_indices]
                rid= rid[valid_indices]
                # 使用索引来子集化每个变量
                img1 = img1[valid_indices]
                # eroded_edge_mask = eroded_edge_mask[valid_indices]
                y1 = y1[valid_indices]
                img2 = img2[valid_indices]
                y2 = y2[valid_indices]
                age1=age1[valid_indices]
                age2=age2[valid_indices]
                different_j_index = np.array(different_j_index[valid_indices])
                mixed_img = np.array(list(mixed_img))[valid_indices]
                mixed_age = mixed_age[valid_indices]

                # different_rid_index = [different_rid_index[i] for i in valid_indices]
                model.eval()
                k = k + len(y1)
                with torch.no_grad():
                    age_diff = (y2.float() - y1.float())# 添加一个维度，形状变为 (batch_size, 1)
                    # 重复 256 次
                    structure1, state1, age_hat1 = model.module.encoder(img1.cuda())

                    structure2, state2, age_hat2_ = model.module.encoder(img2.cuda())
                    if structure_plot:
                        idx_list.append(rid)
                        idx_list.append(rid)
                        structure_list.append(structure1[-2].cpu().detach().numpy())
                        structure_list.append(structure2[-2].cpu().detach().numpy())
                    predicted_img2 = model.module.decoder(structure1, state1, age_diff.cuda())
                    if fastsufer:
                        import scipy.ndimage
                        TARGET_SHAPE = (256, 256, 256)
                        for i in range(predicted_img2.shape[0]):  # 处理 batch 里的每个 MRI
                            gen_img = np.pad(scipy.ndimage.zoom(np.pad(predicted_img2[i].cpu().numpy().squeeze(0) * 255, 4), (220/128, 220/128, 220/128), order=1), 18).astype(np.uint8)


                            orig_img = np.pad(scipy.ndimage.zoom(np.pad(img2[i].cpu().numpy().squeeze(0) * 255, 4), (220/128, 220/128, 220/128), order=1), 18).astype(np.uint8)
                            old_img =  np.pad(scipy.ndimage.zoom(np.pad(img1[i].cpu().numpy().squeeze(0) * 255, 4), (220/128, 220/128, 220/128), order=1), 18).astype(np.uint8)
                            if np.any(gen_img < 0):
                                print("error")
                                print(np.min(gen_img))
                            # 生成带有原始 MRI 头信息的 NIfTI 文件
                            # 参考 NIfTI 头文件（从原始 MRI 提取）
                            REFERENCE_NIFTI = mixed_img[i]
                            reference_nii = nib.load(REFERENCE_NIFTI)
                            reference_nii = nib.Nifti1Image(reference_nii.get_fdata(), affine=reference_nii.affine)
                            new_header = reference_nii.header.copy()
                            # new_header['pixdim'][0] = 1.0

                            # 🚀 Step 4: 生成 NIfTI 文件
                            gen_nii = nib.Nifti1Image(gen_img, affine=reference_nii.affine, header=new_header)
                            orig_nii = nib.Nifti1Image(orig_img, affine=reference_nii.affine, header=new_header)
                            old_nii = nib.Nifti1Image(old_img, affine=reference_nii.affine, header=new_header)

                            # 文件名
                            patient_id = f"patient_{rid[i]}"
                            gen_path = os.path.join(GEN_DIR, f"{patient_id}_{different_j_index[i]}_t1.nii.gz")
                            orig_path = os.path.join(ORIG_DIR, f"{patient_id}_{different_j_index[i]}_t1.nii.gz")
                            old_path = os.path.join(OLD_DIR, f"{patient_id}_{different_j_index[i]}_t1.nii.gz")
                            # 保存文件
                            nib.save(gen_nii, gen_path)
                            nib.save(orig_nii, orig_path)
                            nib.save(old_nii, old_path)

                            subject_list.write(f"generated/{patient_id}_{different_j_index[i]}\n")
                            clean_subject_list.write(f"generated/{patient_id}_{different_j_index[i]}\n")
                            subject_list.write(f"old/{patient_id}_{different_j_index[i]}\n")
                            subject_list.write(f"original/{patient_id}_{different_j_index[i]}\n")
                            clean_subject_list.write(f"original/{patient_id}_{different_j_index[i]}\n")
                            clean_subject_list.write(f"old/{patient_id}_{different_j_index[i]}\n")
                    if any(F.mse_loss(img1.cuda(), predicted_img2, reduction="none").mean(dim=(1, 2, 3,4)) .cpu().numpy()>0.0005):
                        print("i")
                    structure2, state2, age_hat2_o = model.module.encoder(predicted_img2)
                    # structure_mix, state_mix, age_hat_mix = model.module.encoder(mixed_img.cuda())
                age_hat2__=[]
                mse_list = []
                if extra:
                    for mul in range(-11, 11):
                        # 创建 age_gap，并确保形状与模型兼容
                        age_gap = torch.full((len(y1), 1), mul, dtype=torch.float32).cuda()
                        model.eval()

                        with torch.no_grad():
                            # 模型前向传播
                            predicted_img2_ = model.module.decoder(structure1, state1, age_gap)
                            if extra_3d:
                                for i in range(predicted_img2.shape[0]):
                                    gen_img=np.pad(scipy.ndimage.zoom(np.pad(predicted_img2[i].cpu().numpy().squeeze(0) * 255, 4),
                                                          (220 / 128, 220 / 128, 220 / 128), order=1), 18).astype(
                                    np.uint8)
                                    REFERENCE_NIFTI = mixed_img[i]
                                    reference_nii = nib.load(REFERENCE_NIFTI)
                                    reference_nii = nib.Nifti1Image(reference_nii.get_fdata(),
                                                                    affine=reference_nii.affine)
                                    new_header = reference_nii.header.copy()
                                    # new_header['pixdim'][0] = 1.0

                                    # 🚀 Step 4: 生成 NIfTI 文件
                                    gen_nii = nib.Nifti1Image(gen_img, affine=reference_nii.affine, header=new_header)
                                    patient_id = f"patient_{rid[i]}"
                                    gen_path = os.path.join(GEN_DIR, f"{patient_id}_{different_j_index[i]}_{mul+10}_t1.nii.gz")
                            mse_list.append(F.mse_loss(img1.cuda(), predicted_img2_, reduction="none").mean(
                                dim=(1, 2, 3, 4)).cpu().numpy())
                            structure2, state2, age_hat2 = model.module.encoder(predicted_img2_)
                            age_hat2__.append(age_hat2.cpu().detach())
                    fig, ax = plt.subplots(len(age_hat2__[0]), 1, figsize=(10, 15), sharex=True)
                    del structure1, structure2, state2, predicted_img2_
                    torch.cuda.empty_cache()
                    # Reshape mse_list for plotting
                    mse_array = np.array(mse_list).T
                    mse_list_.append(mse_array)
                    age_hat_list.append(age_hat2__)
                    # for i in range(len(age_hat2__[0])):
                    #     ax[i].scatter(range(-11, 11), mse_array[i])
                    #     ax[i].set_ylabel("MSE")
                    #     ax[i].set_title(f"subjects {i + 1}")

                    # # Set common x-axis label
                    # plt.xlabel("age_gap")
                    # plt.tight_layout()
                    # plt.show()
                    # Generate example data: a list of np arrays of length n
                    n = len(age_hat2_[0])  # Length of each np array

                    # Set up the plot
                    # fig, ax = plt.subplots(n, 1, figsize=(10, 15), sharex=True)
                    # list_length = len(age_hat2_)
                    # # Plot each of the n arrays
                    # for i in range(n):
                    #     y_values = [arr[i].cpu() for arr in age_hat2_]  # Extract the i-th value from each array
                    #     ax[i].scatter(range(list_length), y_values)
                    #     ax[i].set_ylabel("Regressed Age")
                    #     ax[i].set_title(f"subjects {i + 1}")
                    #
                    # # Set common x-axis label
                    # plt.xlabel("Age_gap")
                    #
                    # # Adjust layout and show the plot
                    # plt.tight_layout()
                    # plt.show()
                mse_losses = []
                batch_size = img1.shape[0]
                age1_.append(age1.float())
                age2_.append(age2.float())
                mixed_age_.append(mixed_age.float())
                age1_hat_.append(age_hat1.float())
                age2_hat_.append( age_hat2_o.float())
                # mixed_age_hat_.append(age_hat_mix.float())
                for i in range(batch_size):
                    mse_loss = F.mse_loss(img1[i].cuda(), img2[i].cuda(), reduction='sum').item()
                    mse_losses.append(mse_loss)
                ind = torch.tensor(mse_losses) < 6000000
                age_loss_direct += F.l1_loss(age_hat2_o[ind, 0], age2[ind].float().cuda(), reduction='sum').item()
                age_loss_direct_regressed += F.l1_loss((age2 - age1).float().cuda(), age_hat2_o[ind, 0] - age_hat1[ind, 0],
                                                       reduction='sum').item()
                age_loss_base += F.l1_loss(age_hat1[ind, 0], age_hat2_[ind, 0],
                                           reduction='sum').item()  # F.l1_loss(age1.cuda()- age2.cuda(),age_hat1[ind,0] - age_hat2_[ind,0], reduction='sum').item()

                rec_loss_direct += F.mse_loss(predicted_img2[ind].cuda(),
                                              img2[ind].cuda().cuda(), reduction='sum').item()
                rec_loss_base += F.mse_loss(img1[ind].cuda().cuda(),
                                            img2[ind].cuda().cuda(), reduction='sum').item()
                age_loss_base_absolute += F.l1_loss(age2[ind].cuda(), age1[ind].cuda(), reduction='sum').item()
                torch.cuda.empty_cache()
                title = []
                for i in range(len(y1)):
                    title.append(str(y1[i].numpy())[0:5] + '_id:' + str(index[i]))
                title2 = []
                for i in range(len(y2)):
                    title2.append(str(y2[i].numpy())[0:5] + '_id:' + str(index[i]))
                generate_images(predicted_img2, title, f"{idx}_predicted", save_dir + "/image/")
                generate_images(img1, title, f"{idx}_input", save_dir + "/image/")
                generate_images(img2, title2, f"{idx}_target", save_dir + "/image/")
                generate_images(torch.abs(img2.cuda() - predicted_img2), title2, f"{idx}_loss",
                                save_dir + "/image/")
                # generate_images(torch.abs(img2 - img1), title2, f"{idx}_gap_2", save_dir + "/image/")
                generate_images(torch.abs(img2 - img1), title2, f"{idx}_gap",
                                save_dir + "/image/")
                # generate_images((torch.abs(img2.cuda() - img2.cuda()) .cuda()).mean(dim=0, keepdims=True),
                #                 title2[0], f"{idx}_loss", save_dir + "/image/")
                del img1, img2, mixed_img, mixed_age, age1, age2, age_hat1
                torch.cuda.empty_cache()
        if structure_plot:
            structure_total = np.vstack(structure_list)
            index_total = np.hstack(idx_list)
            visualize_tsne(np.mean(structure_total, axis=(2, 3, 4)), index_total, epoch=0, save_path=save_dir, str=True)
        if extra:
            save_path = save_dir + "/image/"  # Specify your save path
            os.makedirs(save_path, exist_ok=True)
            # Concatenate along the batch dimension
            mse_matrix = np.concatenate(mse_list_, axis=0)  # Shape: (100, 22)
            # Plot 100 scatter plots
            fig, axes = plt.subplots(10, 10, figsize=(20, 20), sharex=True, sharey=True)
            axes = axes.flatten()
            for i in range(100):
                axes[i].scatter(range(-11, 11), mse_matrix[i], s=10)
                axes[i].set_title(f"Subject {i + 1}", fontsize=8)
                axes[i].set_xlabel("Age Gap", fontsize=6)
                axes[i].set_ylabel("MSE", fontsize=6)
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "all_scatter_plots.png"))
            plt.close()
            # Calculate mean along the 100 dimension
            mse_mean = mse_matrix.mean(axis=0)
            # Plot the overall mean
            plt.figure(figsize=(10, 6))
            plt.plot(range(-11, 11), mse_mean, marker='o')
            plt.title("Overall Mean MSE vs. Age Gap", fontsize=14)
            plt.xlabel("Age Gap", fontsize=12)
            plt.ylabel("Mean MSE", fontsize=12)
            plt.grid(True)
            plt.savefig(os.path.join(save_path, "mean_plot.png"))
            plt.close()

            print(f"Plots saved to {save_path}")
            # Shape: (25, 22, 4)
            all_rows = []  # 存储最终拼接的数据

            for i in range(7):  # 遍历第一维
                for k in range(len(age_hat_list[i][0])):  # 遍历第三维（不定长）
                    row = []  # 用于存储当前行数据
                    for j in range(22):  # 保留第二维
                        if k < len(age_hat_list[i][j]):  # 确保第三维索引合法
                            row.append(age_hat_list[i][j][k])  # 取出第三维的 k-th 元素
                        else:
                            row.append(np.nan)  # 填充 NaN 以对齐
                    all_rows.append(row)  # 追加到最终列表

            # **转换成 NumPy 矩阵**
            age_hat_matrix = np.array(all_rows)[:,:,0]

            print("Final Matrix Shape:", age_hat_matrix.shape)  # 形状 (N, 22) # Shape: (100, 22)
            # Plot 100 scatter plots
            fig, axes = plt.subplots(10, 10, figsize=(20, 20), sharex=True, sharey=True)
            axes = axes.flatten()
            for i in range(100):
                axes[i].scatter(range(22), age_hat_matrix[i], s=10)
                axes[i].set_title(f"Subject {i + 1}", fontsize=8)
                axes[i].set_xlabel("Age Gap", fontsize=6)
                axes[i].set_ylabel("Age Hat", fontsize=6)
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "all_age_hat_scatter_plots.png"))
            plt.close()
            # Calculate mean along the 100 dimension
            age_hat_mean = age_hat_matrix.mean(axis=0)
            # Plot the overall mean
            plt.figure(figsize=(10, 6))
            plt.plot(range(-11, 11), age_hat_mean, marker='o')
            plt.title("Overall Mean Age Hat vs. Age Gap", fontsize=14)
            plt.xlabel("Age Gap", fontsize=12)
            plt.ylabel("Mean Age Hat", fontsize=12)
            plt.grid(True)
            plt.savefig(os.path.join(save_path, "mean_age_hat_plot.png"))
            plt.close()
            print(f"Plots saved to {save_path}")
        print("age loss direct construct: ", age_loss_direct / k)
        print("age loss baseline: ", age_loss_base / k, age_loss_base_absolute / k)
        print("rec loss direct construct: ", rec_loss_direct / k)
        print("rec loss baseline: ", rec_loss_base / k)
        print("age_loss_direct_regressed:", age_loss_direct_regressed / k)
        age1_=torch.concat(age1_,0)
        age2_=torch.concat(age2_,0)
        mixed_age_=torch.concat(mixed_age_,0)
        age_hat1_=torch.concat(age1_hat_,0)
        age_hat2_=torch.concat(age2_hat_,0)
        mixed_age_hat_=torch.concat(mixed_age_hat_,0)
        visualize_age_scatter((age_hat1_[:,0]-age_hat2_[:,0]).cpu(),age1_-age2_, save_path=save_dir + "/image/",epoch=0,
                                             )
        visualize_age_scatter(age_hat1_[:,0].cpu(),age1_, save_path=save_dir + "/image/",epoch=1,
                                             )
        print(k)

    else:

        model_encode.cuda()
        model_decode.cuda()
        # percep = PerceptualLoss(stage1=True)
    # Initialize optimizer
        optimizer = torch.optim.Adam( list(model_encode.parameters()) + list(model_decode.parameters()), lr=LR, betas=(0.5, 0.999))

        # save_dir = '/storage/home/hcoda1/8/yli3863/scratch/FS/model_save/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model, total_loss, image_similarity_loss, structure_vec_sim_loss, age_prediction_loss, semi_result = Pretext(model_encode,model_decode,
                                                                                                                     optimizer,
                                                                                                                     EPOCHS,
                                                                                                                     train_dataset,
                                                                                                                     val_loader,
                                                                                                                     test_loader,
                                                                                                                     STRUCTURE_VEC_SIMILARITY_LOSS_MULT,
                                                                                                                     AGE_LOSS_MULT,
                                                                                                                     save_dir,percep=None)