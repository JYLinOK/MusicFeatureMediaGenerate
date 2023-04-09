import os
import os.path
import time
from PIL import Image


# New Size
new_batch_size = (170, 170)
# Sace dir after change
new_save_dir = './music_jpg170/'
# Previous data dir 
img_data_path = './MusicData/'



# ================================================================
# Main Code
img_data = os.listdir(img_data_path)
print(f'{img_data = }')

if not os.path.exists(new_save_dir): os.mkdir(new_save_dir)


# i=1 put here, namming following the folders
# i = 1

for folder in img_data:
    imgs_l = os.listdir(img_data_path+folder)
    # i=1 put here, namming following the files
    i = 1

    for img in imgs_l:
        imag_path = img_data_path+folder+os.sep+img

        print()
        print(f'{imag_path = }')
        
        imag = Image.open(imag_path).convert('RGB')
        print(f'{imag.size = }')

        save_dir = new_save_dir+folder
        if not os.path.exists(save_dir): os.mkdir(save_dir)

        # Change format
        img_format = '.jpg'
        # Save dir
        save_path = save_dir+os.sep+str(i)+img_format

        imag_resize = imag.resize(new_batch_size)  
        imag_resize.save(save_path)

        i+=1


