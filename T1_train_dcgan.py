from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mynet.modcgan import discriminator, generator
from utils.mydataloader import DCGan_collate_fn, MyDataset
from utils.utils import get_lr_scheduler, set_optimizer_lr
from utils.utils_epoch import train_a_epoch




CUDA = True
iffp16 = True
Gen_model_path = ""
Dis_model_path = ""
 
# Training configuration
# ===================================================================
t_channels = 64
t_input_shape = [170, 170]
t_init_epochs = 0
t_epoch = 100
t_batch_size = 2
t_num_train = 25
t_img_save_steps = 5
t_num_workers = 4
train_sampler = None
shuffle = False
# ===================================================================


t_init_lr = 2e-3
t_min_lr = t_init_lr * 0.01
t_optimizer_type = "adam"
t_momentum = 0.5
t_weight_decay = 0
lr_change_method = "cos"
t_save_period = 10
t_save_dir = 'log'

feature_img_dir = './feature_img'






if __name__ == '__main__':

    feature_img_dir = Path(feature_img_dir)
    feature_img_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    Gen_model = generator(t_channels, t_input_shape)
    Dis_model = discriminator(t_channels, t_input_shape)

    if Gen_model_path != '':
        model_dict = Gen_model.state_dict()
        pretrained_dict = torch.load(Gen_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        Gen_model.load_state_dict(model_dict)

    if Dis_model_path != '':
        model_dict = Dis_model.state_dict()
        pretrained_dict = torch.load(Dis_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        Dis_model.load_state_dict(model_dict)

    BCEL_loss = nn.BCEWithLogitsLoss()

    from torch.cuda.amp import GradScaler as GradScaler
    scaler = GradScaler()

    G_model_train = Gen_model.train()
    D_model_train = Dis_model.train()

    if CUDA:
        cudnn.benchmark = True
        G_model_train = torch.nn.DataParallel(Gen_model)
        G_model_train = G_model_train.cuda()

        D_model_train = torch.nn.DataParallel(Dis_model)
        D_model_train = D_model_train.cuda()

    if True:
        Gen_optimizer = {
            'adam': optim.Adam(G_model_train.parameters(), lr=t_init_lr, betas=(t_momentum, 0.999), weight_decay=t_weight_decay),
            'sgd': optim.SGD(G_model_train.parameters(), t_init_lr, momentum=t_momentum, nesterov=True)
        }[t_optimizer_type]

        Dis_optimizer = {
            'adam': optim.Adam(D_model_train.parameters(), lr=t_init_lr, betas=(t_momentum, 0.999), weight_decay=t_weight_decay),
            'sgd': optim.SGD(D_model_train.parameters(), t_init_lr, momentum=t_momentum, nesterov=True)
        }[t_optimizer_type]

        lr_scheduler_func = get_lr_scheduler(
            lr_change_method, t_init_lr, t_min_lr, t_epoch)

        t_epoch_step = t_num_train // t_batch_size

        annotate_file = './music_lables/data/normal/label.csv'
        train_dataset = MyDataset(
            annotate_file=annotate_file,
            input_shape=t_input_shape
        )

        # a = train_dataset.__getitem__(0)
        # print(f'{a = }')
  

        dataL = DataLoader(
            dataset=train_dataset,
            batch_size=t_batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=t_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=DCGan_collate_fn
        )


        # Start the trainning
        for epoch in range(t_init_epochs, t_epoch):

            set_optimizer_lr(Gen_optimizer, lr_scheduler_func, epoch)
            set_optimizer_lr(Dis_optimizer, lr_scheduler_func, epoch)

            train_a_epoch(
                G_model_train,
                D_model_train,
                Gen_model,
                Dis_model,
                Gen_optimizer,
                Dis_optimizer,
                BCEL_loss,
                epoch,
                t_epoch_step,
                dataL,
                t_epoch,
                CUDA,
                iffp16,
                scaler,
                t_save_period,
                t_save_dir,
                t_img_save_steps,
                local_rank,
                feature_img_dir
            )







