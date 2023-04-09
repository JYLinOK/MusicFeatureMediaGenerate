
import os

from trans import trans_content


# Transform configuration
# ===================================================================
original_img_dir = './music_jpg170'
feature_img_dir = './feature_img'
feature_trans_dir = './feature_trans'
# ===================================================================

if not os.path.exists(feature_trans_dir): os.mkdir(feature_trans_dir)



def feature2trans(original_img_dir, feature_img_dir, feature_trans_dir, alpha=0.6):
    original_dir_list = os.listdir(original_img_dir)
   
    for dir in original_dir_list:
        print()
        print(f'{dir = }')

        for dir_file in os.listdir(os.path.join(original_img_dir, dir)):
            original_img_path = os.path.join(original_img_dir, dir, dir_file)
            print(f'\n{original_img_path = }')

            feature_img_path = os.path.join(feature_img_dir, dir, dir_file)
            print(f'{feature_img_path = }')

            trans_content(original_img_path, feature_img_path, feature_trans_dir)




if __name__ == '__main__':
    feature2trans(original_img_dir, feature_img_dir, feature_trans_dir)

    


    



