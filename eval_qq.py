import os
import sys
sys.path.append('/home/ubuntu/YourPath')  
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
from models.aesclip import AesCLIP,AesCLIP_reg
from dataset import AVA
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
from sklearn.metrics import mean_squared_error

import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--path_to_images', type=str, default='./APDDv2image',
                        help='directory to images')

    parser.add_argument('--path_to_save_csv', type=str,
                        default="./csvfiles",
                        help='directory to csv_folder')

    parser.add_argument('--experiment_dir_name', type=str,
                        default='./result/ava/',
                        help='directory to project')

    parser.add_argument('--batch_size', type=int, default=64, help='how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers',
                        )
    args = parser.parse_args()
    return args


opt = init()


def adjust_learning_rate(params, optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = params.init_lr * (0.5 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_score(opt, y_pred):
    score_np = y_pred.data.cpu().numpy()
    return y_pred, score_np

def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_save_csv, 'train.csv')
    test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')

    train_ds = AVA(train_csv_path, opt.path_to_images, if_train=True)
    test_ds = AVA(test_csv_path, opt.path_to_images, if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, test_loader



def validate(opt, model, loader):
    model.eval()
    model.cuda()
    total_pred_scores = []
    total_true_scores = []
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.cuda()
        y = y.type(torch.FloatTensor)
        y = y.cuda()
        with torch.no_grad():
            y_pred = model(x)


        pred_score, pred_score_np = get_score(opt, y_pred)
        true_score, true_score_np = get_score(opt, y)

        total_pred_scores.append(pred_score_np)
        total_true_scores.append(true_score_np)


    total_pred_scores = np.concatenate(total_pred_scores)
    total_true_scores = np.concatenate(total_true_scores)
    mse = mean_squared_error(total_true_scores, total_pred_scores)
    srocc, _ = spearmanr(total_true_scores, total_pred_scores)
    wasd = np.mean(np.abs(total_true_scores - total_pred_scores))

    print(f'WASD: {wasd}, MSE: {mse}, SROCC: {srocc}')



    lcc_mean = pearsonr(total_pred_scores, total_true_scores)
    print('PLCC', lcc_mean[0])

    threshold = 5.0
    total_true_scores_lable = np.where(total_true_scores <= threshold, 0, 1)
    total_pred_scores_lable = np.where(total_pred_scores <= threshold, 0, 1)
    acc = accuracy_score(total_true_scores_lable, total_pred_scores_lable)
    print('ACC', acc)

def add_prefix_to_state_dict(state_dict, prefix):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = prefix + k
        new_state_dict[new_key] = v
    return new_state_dict

def start_eval(opt):
    train_loader, test_loader = create_data_part(opt)

    model = AesCLIP_reg(clip_name='ViT-B/16', weight='./modle_weights/0.ArtCLIP_weight--e11-train2.4314-test4.0253_best.pth')
    model.load_state_dict(torch.load('./modle_weights/1.Score_reg_weight--e4-train0.4393-test0.6835_best.pth'))   ####Modifying the attribute scoring model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model = model.cuda()
    validate(opt, model=model, loader=test_loader)

if __name__ == "__main__":
    start_eval(opt)