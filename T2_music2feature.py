
import os
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from utils.mydataloader import MyDatasetFea


# Transform configuration
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
feature_img_dir = './feature_img'
annotate_file = './music_lables/data/normal/label.csv'
# ===================================================================


if not os.path.exists(feature_img_dir): os.mkdir(feature_img_dir)



def music2feature(dataL):

    for feature_img, img_path in dataL:
        print(f'\n{len(feature_img) = }')
        print(f'{len(img_path) = }')    

        feature_imgs, img_paths = feature_img, img_path
        feature_imgs_np = feature_imgs.numpy()

        print(f'{img_paths = }')


        # ==================================================
        for i in range(len(feature_imgs_np)):
            img_i = feature_imgs_np[i]*255
            img_i = Image.fromarray((img_i.transpose(1, 2, 0)).astype(np.uint8))    

            img_pre_path = img_paths[i]
            img_name = os.path.split(img_pre_path)[1]
            img_dir = os.path.basename(os.path.dirname(img_pre_path))   

            print(f'{img_name = }')
            print(f'{img_dir = }')
            print(f'{os.path.basename(img_pre_path) = }')
            print() 

            img_dir = os.path.join(feature_img_dir, img_dir,)
            if not os.path.exists(img_dir): os.mkdir(img_dir)
            print(f'{img_dir = }')  

            feature_img_path =  os.path.join(img_dir, img_name)
            print(f'{feature_img_path = }') 

            img_i.save(feature_img_path)
        # ==================================================    




train_dataset = MyDatasetFea(
    annotate_file=annotate_file,
    input_shape=t_input_shape
)

dataL = DataLoader(
            dataset=train_dataset,
            batch_size=t_batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=t_num_workers,
            pin_memory=True,
            drop_last=True,
        )



if __name__ == '__main__':
    music2feature(dataL)

    


    



