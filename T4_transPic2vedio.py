
import os

import cv2
import librosa

from moviepy.editor import VideoFileClip, AudioFileClip


# Transform configuration
# ===================================================================
feature_trans_dir = './feature_trans'
piano_music_dir = './music_piano/origin'
feature_video_dir = './feature_video'
video_music_dir = './feature_v_m'
# ===================================================================

if not os.path.exists(feature_video_dir): os.mkdir(feature_video_dir)
if not os.path.exists(feature_video_dir): os.mkdir(feature_video_dir)
if not os.path.exists(video_music_dir): os.mkdir(video_music_dir)


def gen_video(dir, imgs_dir, video_path):
    print()
    music_path = piano_music_dir     + '/' + dir+ '.mp3'
    print(f'{music_path = }')

    music_duration = librosa.get_duration(path=music_path)
    print(f'{music_duration = }')

    print(f'2: {imgs_dir = }')
    imgs_list = os.listdir(imgs_dir)
    print(f'{len(imgs_list) = }')

    fps = int(music_duration/len(imgs_list))
    fps = 1/fps
    print(f'{fps = }')

    dpi = [1920, 1080]
    v_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    v_writer = cv2.VideoWriter(video_path, v_fourcc, fps, dpi)

    for img_i in imgs_list:
        img_path = os.path.join(imgs_dir, img_i)
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, dpi)
        v_writer.write(frame)

    v_writer.release()



def combine_video_music(video_dir, music_dir, video_music_dir):
    video_list = os.listdir(video_dir)
    # print(f'\n{video_list}')
    for video in video_list:
        video_path = os.path.join(video_dir, video)

        print(f'{video_path = }')
        music_path = os.path.join(music_dir, video.split('.')[0]+'.mp3')
        video_fc = VideoFileClip(video_path)

        videos_com = video_fc.set_audio(AudioFileClip(music_path))
        videos_com_path = os.path.join(video_music_dir, video)
        videos_com.write_videofile(videos_com_path, audio_codec='aac')



def transPic2video(feature_trans_dir, feature_video_dir, piano_music_dir, video_music_dir):
    feature_trans_dir_list = os.listdir(feature_trans_dir)
   
    # for dir in feature_trans_dir_list:
    #     print()
    #     print(f'{dir = }')

    #     imgs_dir = os.path.join(feature_trans_dir, dir)
    #     print(f'{imgs_dir = }')

    #     piano_video_path = os.path.join(feature_video_dir, str(dir)+'.mp4')
    #     print(f'{piano_video_path = }')

    #     gen_video(dir, imgs_dir, piano_video_path)
    
    combine_video_music(feature_video_dir, piano_music_dir, video_music_dir)




if __name__ == '__main__':
    transPic2video(feature_trans_dir, feature_video_dir, piano_music_dir, video_music_dir)

    


    



