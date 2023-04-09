import os
import ffmpeg
import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.silence import detect_silence, detect_nonsilent
from pydub.utils import make_chunks
import librosa
import math
import torch 
import numpy as np
import csv
from cv2 import imread, imwrite

import musicUtils


origin_piano_dir = './music_piano/origin'
nonoise_piano_dir = './music_piano/nonoise'
split_piano_dir = './music_piano/splited'
if not os.path.exists(origin_piano_dir): os.mkdir(origin_piano_dir)
if not os.path.exists(nonoise_piano_dir): os.mkdir(nonoise_piano_dir)
if not os.path.exists(split_piano_dir): os.mkdir(split_piano_dir)


# _________________________________________________________________________________________________________________
# Split the piano songs into multiple small ones
preprocess_spliting = True
if not preprocess_spliting:
    origin_piano_dir = './music_piano/origin'
    for piano in os.listdir(origin_piano_dir):
        split_dir2 = split_piano_dir + '/' + piano.split('.')[0]
        piano2 = origin_piano_dir + '/' + piano
        print(f'\n{split_dir2 = }')
        print(f'{piano2 = }')
        # splitMusicBySilence(music, split_piano_dir)
        epoch = 26
        musicUtils.splitMusicByTime(piano2, split_dir2, epoch=epoch-1)

        

# _________________________________________________________________________________________________________________
# Reduce the noise of each song on the dir
preprocess_nonoise = True
if not preprocess_nonoise:
    for piano_dir in os.listdir(split_piano_dir):
        for pinao_music in os.listdir(split_piano_dir+'/'+piano_dir):
            reduced_noise_song_np, sr = musicUtils.ruduce_noise(split_piano_dir+'/'+piano_dir+'/'+pinao_music)
            new_dir_path = nonoise_piano_dir+'/'+piano_dir
            new_file_path = new_dir_path+'/'+pinao_music.split('.')[0]+'.wav'
            if not os.path.exists(new_dir_path): os.mkdir(new_dir_path)
            musicUtils.saveNpaAsMusic(new_file_path, sr=sr, npa=reduced_noise_song_np)



# _________________________________________________________________________________________________________________
def get_N_splited_npList(npa, each_len, n):
    '''
    return splited numpy sub-array list with fixed length
    '''
    return [npa[each_len*i:each_len*(i+1)] for i in range(n)]


def conv_2npa(np1, np2, stribe=1):
    '''
    return the convolution result of two numpy arrays
    '''
    t_a = torch.reshape(torch.from_numpy(np1).float(), [1, 1]+list(np1.shape))
    t_b = torch.reshape(torch.from_numpy(np2).float(), [1, 1]+list(np2.shape))
    conv = torch.conv2d(t_a, t_b, stride=stribe)
    return conv.numpy()


def normal_npa(size=(12, 12)):
    '''
    return a normal npa with the item range from 0 to 1
    '''
    rand_npa = np.random.randn(size[0], size[1])
    np_new = rand_npa-np.min(rand_npa)+1
    return np_new/np.max(np_new)


def uniform_npa(size=(12, 12)):
    '''
    return an uniform npa with the item range from 0 to 1
    '''
    return np.random.uniform(0, 1, size)


# Generate the label files
generate_label = True

if not generate_label:
    music_root = './music_piano/nonoise'
    len_list = []

    # First: get the minimum length of each splited music fragment
    for dir in os.listdir(music_root):
        print(f'{dir = }')
        for music in os.listdir(os.path.join(music_root, dir)):
            if music.split('.')[0] != '25':
                img_path = os.path.join(music_root, dir, music)
                npa, sr = musicUtils.music2np(img_path)
                len_list.append(len(npa))

    customized_len = True
    if not customized_len:
        # Same average length: min_piano_frag_len = 3659.44
        print(f'\n{min(len_list) = }')  # min(len_list) = 91486
        min_piano_frag_len = int(min(len_list)/3)
        rec_len = int(math.sqrt(min_piano_frag_len))
        min_piano_frag_len = 3*rec_len*rec_len
        print(f'\n{min_piano_frag_len = }')  # min_piano_frag_len = 90828
    else:
        min_piano_frag_len = 3*170*170  # 86700

    # _______________________________________
    # generate the convolution kernel
    normal_kernel = True
    # normal_kernel = False
    # _______________________________________

    label_csv_path = './music_lables/data/'
    label_kernel_csv_path = './music_lables/kernel/'
    if not os.path.exists(label_csv_path): os.mkdir(label_csv_path)
    if not os.path.exists(label_kernel_csv_path): os.mkdir(label_kernel_csv_path)
    
    kernel_size = (12, 12)
    if normal_kernel: 
        kernel = normal_npa()
        sub_dir = 'normal'
    else: 
        kernel = uniform_npa()
        sub_dir = 'uniform'

    csv_path = os.path.join(label_csv_path, sub_dir)
    kernel_csv_path = os.path.join(label_kernel_csv_path, sub_dir)

    if not os.path.exists(csv_path): os.mkdir(csv_path)
    if not os.path.exists(kernel_csv_path): os.mkdir(kernel_csv_path)

    with open(os.path.join(kernel_csv_path, 'kernel.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(kernel.tolist())

    # Second: make a traversal to get the label data for each picture.
    with open(os.path.join(csv_path, 'label.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['picture', 'feature'])

        for dir in os.listdir(music_root):
            print(f'{dir = }')
            wav_path = os.path.join(music_root, dir)
            for wav_i in os.listdir(wav_path):
                number = wav_i.split('.')[0]
                print(f'\n{number = }')

                if number != '25':
                    music_path = os.path.join(wav_path, wav_i)
                    print(f'{music_path = }')

                    npa, sr = musicUtils.music2np(music_path)

                    npa = npa[0:min_piano_frag_len]
                    print(f'{npa.shape = }')

                    conved_rect_npa = npa

                    # rect_npa = npa.reshape(a_len, a_len)
                    # print(f'{rect_npa.shape = }')
                    # conved_rect_npa = conv_2npa(rect_npa, kernel, stribe=5)
                    # conved_rect_npa = conved_rect_npa[0][0].flatten()
                    # print(f'{conved_rect_npa.shape = }')

                    conved_rect_npa = conved_rect_npa.tolist()
                    
                    pic_path = os.path.join('./music_jpg', dir, str(int(number)+1)+'.jpg')
                    row_data = [pic_path,  conved_rect_npa]
                    writer.writerow(row_data)

                    print(f'{type(conved_rect_npa[0]) = }')
                    print(f'{len(conved_rect_npa) = }')


      
# _________________________________________________________________________________________________________________
# Change png to jpg
change_png_to_jpg = True
if not change_png_to_jpg:
    jpg_dir = './music_jpg'
    png_dir = './music_pictures'
    if not os.path.exists(jpg_dir): os.mkdir(jpg_dir)
    for folder in os.listdir(png_dir):
        for png_img in os.listdir(os.path.join(png_dir, folder)):
            png_img_path = os.path.join(png_dir, folder, png_img)
            png_img_data = imread(png_img_path)
            jpg_img_dir = os.path.join(jpg_dir, folder)
            if not os.path.exists(jpg_img_dir): os.mkdir(jpg_img_dir)
            jpg_img_path = os.path.join(jpg_img_dir, png_img.replace('png', 'jpg'))
            imwrite(jpg_img_path, png_img_data)

























                




