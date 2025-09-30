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

# ======= Path Configuration =======
# Base data path - modify this according to your environment
BASE_PATH = "../../../pub/liyifan/mri"

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
parser.add_argument('--save_dir', type=str, default=BASE_PATH + '/FS/model_save/',
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
    global reserved_memories

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible_devices:
            print("No GPUs are set in CUDA_VISIBLE_DEVICES.")
            return

        gpu_indices = list(map(int, visible_devices.split(",")))
        current_device = torch.cuda.current_device()

        for i, gpu_id in enumerate(gpu_indices):
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            reserved_memory = int(total_memory * reserve_ratio)

            try:
                tensor = torch.ones((reserved_memory // 4,), dtype=torch.float32, device=f"cuda:{i}", requires_grad=False)
                reserved_memories.append(tensor)

                print(f"Reserved {reserved_memory / 1024**3:.2f} GB on cuda:{i} (Physical GPU {gpu_id}).")

            except RuntimeError:
                print(f"Warning: Could not reserve full memory on cuda:{i} (Physical GPU {gpu_id}), possibly out of memory.")

        torch.cuda.set_device(current_device)

        print("GPU memory successfully reserved and is available for reuse.")
    del reserved_memories


class PerceptualLoss(nn.Module):
    def __init__(self,
                 stage1=True,
                 pretrained_path="./resnet_10.pth"):
        super(PerceptualLoss, self).__init__()
        self.stage1 = stage1

        if stage1 is not None:
            from MedicalNet.models.resnet import resnet10
        else:
            self.net_3d = None

    def forward(self,
                target_img,
                input_img,
                model,
                sturcture_n1,
                state_n1,
                age,
                state="train",
                stage1=None):
        if stage1 is not None:
            self.stage1 = stage1

        if self.stage1 is None:
            return torch.tensor(0.0, device=input_img.device)

        if self.stage1:

            rec_loss = F.mse_loss(input_img, target_img, reduction="mean")

            return  rec_loss * 500#+loss_perc

        else:
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
            # feat_in = self.net_3d(input_img)
            # feat_tar = self.net_3d(target_img)

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
        plt.figure(figsize=(10, 8))
        unique_ages = np.unique(clinical_age1_)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', '*', 'X', 'h']
        num_shapes = len(markers)
        colors = [plt.get_cmap('tab10')(i / len(unique_ages)) for i in range(len(unique_ages))]

        for i, age in enumerate(unique_ages):
            age_mask = np.array(clinical_age1_) == age
            plt.scatter(
                feature_tsne[age_mask, 0],
                feature_tsne[age_mask, 1],
                c=[colors[i]],
                label=f'id {age}',
                marker=markers[i % num_shapes],
                s=100
            )

        plt.legend()
        file_name = f'epoch_{epoch}_str_train.png'
    else:
        file_name = f'epoch_{epoch}_age_train.png'

        norm = plt.Normalize(clinical_age1_.min(), clinical_age1_.max())

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=clinical_age1_, cmap='viridis', norm=norm)

        plt.colorbar(scatter, label='Clinical Age')

        plt.title(f"t-SNE Visualization of Features with Clinical Age Color Coding - Epoch {epoch}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    file_path = save_path +'/'+ file_name
    plt.savefig(file_path)

    plt.close()

    return file_path
def visualize_age_scatter(age_, clinical_age1_, epoch, save_path='./'):
    plt.figure(figsize=(8, 6))
    plt.scatter(age_, clinical_age1_, c='b', marker='o', alpha=0.5)
    plt.xlabel('Age (Predicted)')
    plt.ylabel('Absolute Age')
    plt.title(f'Scatter plot of Age vs Absolute Age (Epoch {epoch})')

    save_file = f'{save_path}/epoch_{epoch}_age_scatter.png'
    plt.savefig(save_file)
    plt.close()


def visualize_age_matrix_scatter(matrix1, matrix2, epoch, save_path='./',head="original_gap"):
    slice1 = matrix1[:, 7, :, :].cpu().detach().numpy()
    slice2 = matrix2[:, 7, :, :].cpu().detach().numpy()

    fig, axes = plt.subplots(15, 15, figsize=(20, 20))

    for i in range(15):
        for j in range(15):
            ax = axes[i, j]
            ax.scatter(slice1[:, i, j], slice2[:, i, j], c='b', alpha=0.5, s=5)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(head+f'(_Epoch {epoch})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

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
            age_diff = (y1.float() - y2.float())

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
            loss_rec = recon_loss
            if epoch<stage1_epoch+stage0_epoch:
                loss = loss_rec+  loss_age
                wandb.log({"train_stage0/loss_age_gap": loss_age_gap, "train_stage0/loss_age_mean": loss_age_mean,
                           "train_stage0/loss_rec": loss_rec
                           })
            else:
                loss = loss_rec

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
                visualize_age_scatter( age_hat1[:, 0].cpu().detach().numpy(),y1.cpu().detach().numpy(),
                                      str(epoch) + 'predicted_aging_index_vs_actual', save_path=save_dir)
            loss_structure = trip_c
            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_encode.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(model_decode.parameters(), max_norm=0.5)
            optimizer.step()

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

            age_diff = (y1.float() - y2.float())
            loss_age.append( ((torch.norm(y1.cuda() - age_hat1[:,0]) ** 2 )/ ( y1.shape[0])).cpu().detach().numpy())

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

            age_diff = (y1.float() - y2.float())
            age_hat1 = age_hat1.cpu().numpy().flatten()
            age_hat2 = age_hat2.cpu().numpy().flatten()
            y1 = y1.numpy().flatten()
            y2 = y2.numpy().flatten()
            ages['y'].extend(y1)
            ages['y'].extend(y2)
            ages['age_hat'].extend(age_hat1)
            ages['age_hat'].extend(age_hat2)

            predicted_img1=model_decode(structure2,state2, age_diff.cuda())

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

    predicted_imgs = predicted_imgs.cpu().detach().numpy()
    batch_size = predicted_imgs.shape[0]
    img_height, img_width = resize_dim,resize_dim


    hseq = []

    for i in range(batch_size):
        vseq = []

        img = predicted_imgs[i,0, :, :, 60]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        age_text = f"Age: {age1[i]}"
        cv2.putText(img, age_text, (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.2, (255, 255, 255), 1, cv2.LINE_AA)

        vseq.append(img)

        vseq = np.vstack(vseq)
        hseq.append(vseq)

    hseq = np.hstack(hseq)

    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, f"{image_name}.png")
    cv2.imwrite(save_path, hseq)


if __name__ == "__main__":
    occupy_gpu_memory()

    # Model save path
    save_path = BASE_PATH + "/3dunet"
    save_path_ = os.path.join(save_path, "ex_{}".format(
        datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    ))

    wandb.init(
        project="3dUnet_255_all_flip",
        config={
            "save_path": save_path_,
            "architecture": "train with val",
            "resume": BASE_PATH + "/3dunet/ex_2025_02_26_08-43-39/check_point_model_290_1_1.pth",
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs
        }
    )
    
    # Load pretrained model
    path_model = BASE_PATH + "/3dunet/ex_2025_02_26_08-43-39/check_point_model_290_1_1.pth"
    if path_model is not None:
        state_dict = torch.load(path_model)
        new_state_dict = {k.replace('.module.', '.'): v for k, v in state_dict.items()}

        from ae_3d import UNetDecoder3D, UNetEncoder3D


        image_size = 128
        model_encode = nn.DataParallel(UNetEncoder3D())
        model_decode = nn.DataParallel(UNetDecoder3D())
        encoder_state_dict = {k.replace("encoder.", ""): v for k, v in new_state_dict.items() if k.startswith("encoder.")}
        decoder_state_dict = {k.replace("decoder.", ""): v for k, v in new_state_dict.items() if k.startswith("decoder.")}

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

    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Dataset paths
    train_dir = BASE_PATH + '/FS_2/processed_health_all/train_val/'
    test_dir = BASE_PATH + '/FS_2/processed_health_all/test/'
    val_dir = BASE_PATH + '/FS_2/processed_health_all/validation/'
    train_dataset= MRIDataset( train_dir, resize_dim,train=True)
    train_inference = MRIDataset_inference(train_dir, resize_dim)


    val_loader = torch.utils.data.DataLoader(MRIDataset(val_dir, resize_dim, train=False),
                                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(MRIDataset(test_dir, resize_dim, train=False),
                                              batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Start training
    model_encode.cuda()
    model_decode.cuda()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(list(model_encode.parameters()) + list(model_decode.parameters()), 
                                 lr=LR, betas=(0.5, 0.999))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Train model
    model, total_loss, image_similarity_loss, structure_vec_sim_loss, age_prediction_loss, semi_result = Pretext(
        model_encode, model_decode,
        optimizer,
        EPOCHS,
        train_dataset,
        val_loader,
        test_loader,
        STRUCTURE_VEC_SIMILARITY_LOSS_MULT,
        AGE_LOSS_MULT,
        save_dir, percep=None)

    print("Training completed!")
 