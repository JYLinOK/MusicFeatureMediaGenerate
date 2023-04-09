import os
import os.path
import time
from PIL import Image



new_batch_size = (50, 50)
img_data_path = './music_pictures/'
new_save_dir = './new_music_pictures/'

img_data = os.listdir(img_data_path)
print(f'{img_data = }')

if not os.path.exists(new_save_dir): os.mkdir(new_save_dir)

for folder in img_data:
    imgs_l = os.listdir(img_data_path+folder)
    for img in imgs_l:
        imag_path = img_data_path+folder+os.sep+img

        print()
        print(f'{imag_path = }')
        
        imag = Image.open(imag_path) 
        print(f'{imag.size = }')

        save_dir = new_save_dir+folder
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        save_path = save_dir+os.sep+img
        imag_resize = imag.resize(new_batch_size)  
        imag_resize.save(save_path)



