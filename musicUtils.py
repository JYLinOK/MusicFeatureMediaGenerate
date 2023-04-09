import librosa
import numpy as np
import os
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import librosa.display
import scipy.io.wavfile
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
import noisereduce as nr
import scipy
from scipy import io



def music2np(music_path):
    '''
    change music to numpy array    
    '''
    na, sr = librosa.load(music_path)
    return na, sr


def saveNpaInCSV(npa, csv_path):
    '''
    save numpy array in CSV file
    '''
    np.savetxt(csv_path, npa, delimiter=',')


def npaFromCSV(csv_path):
    '''
    read numpy array from CSV file
    '''
    npa = np.genfromtxt(csv_path, defaultfmt=',')
    return npa


def showWaveGraph(music_path, w=10, h=5):
    '''
    show the wave graph of music  
    '''
    y, sr = music2np(music_path)
    plt.figure(figsize=(w, h))
    librosa.display.waveshow(y, sr=sr)
    plt.show()


def showSpec(npa, sr, x_axis='time', y_axis='hz', w=10, h=5):
    '''
    show the Spec graph of music
    '''
    plt.figure(figsize=(w, h))
    librosa.display.specshow(npa, sr=sr, x_axis=x_axis, y_axis=y_axis)
    plt.colorbar()
    plt.show()


def showSTFTGraph(music_path, w=10, h=5):
    '''
    show the STFT transoformed graph of music
    '''
    y, sr = music2np(music_path)
    y_stft = librosa.stft(y)
    y_stft_db = librosa.amplitude_to_db(abs(y_stft))
    showSpec(y_stft_db, sr, x_axis='time', y_axis='hz', w=w, h=h)


def showMFCCGraph(music_path):
    '''
    show the MFCC graph of music
    '''
    y, sr = music2np(music_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc


def getMFCCGraph(music_path):
    '''
    get the MFCC graph of music
    '''
    y, sr = music2np(music_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc


def showZCRGraph(music_path, w=10, h=5):
    '''
    show the ZCR graph of music
    '''
    y, sr = music2np(music_path)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    plt.figure(figsize=(w, h))
    plt.plot(zcr[0])
    plt.show()
    return zcr


def showSpectralCentroid(music_path, w=10, h=5):
    '''
    show the spectral centroid graph of music
    '''
    def normalize(y, axisn=0):
        return minmax_scale(y, axis=axisn)

    y, sr = music2np(music_path)
    sp_cen = librosa.feature.spectral_centroid(y=y, sr=sr)

    frames = range(len(sp_cen))
    t = librosa.frames_to_time(frames)

    plt.figure(figsize=(w, h))
    plt.plot(t, normalize(sp_cen), color='blue')
    plt.show()
    # return sp_cen
    return normalize(sp_cen)


def saveNpaAsMusic(music_name, sr, npa):
    '''
    write numpy data into a music file, e.g.: mp3 or wav
    '''
    scipy.io.wavfile.write(music_name, sr, npa)



def splitMusicBySilence(music, split_dir, format='mp3', min_len=500, thresh=2):
    '''
    Split the music by special silence time
    '''
    sound = AudioSegment.from_file(music, format=format)

    # print the avarage loudness of the sound
    print(f'{sound.dBFS = }')
    sounds_split = split_on_silence(sound, min_silence_len=min_len, silence_thresh=sound.dBFS-thresh)
    sounds_split_son_dir = split_dir+'/'+music.split('.')[0]
    if not os.path.exists(split_dir): os.mkdir(split_dir)
    if not os.path.exists(sounds_split_son_dir): os.mkdir(sounds_split_son_dir)

    for i, sub_music in enumerate(sounds_split):
        sub_music.export(sounds_split_son_dir+'/'+str(i)+'.'+format, format=format)


def splitMusicByTime(music, split_dir, epoch, size_ms=2000,  format='mp3'):
    '''
    Split the music by special time
    epoch != -1 >> automatically allocate the split size_ms 
    '''
    sound = AudioSegment.from_file(music, format=format)

    # print the avarage loudness of the sound
    print(f'{sound.dBFS = }')
    sound_len = len(sound)
    print(f'{sound_len = }')

    if epoch != -1: size_ms = int(sound_len/epoch)

    chunks_split = make_chunks(sound, size_ms)
    chunks_split_son_dir = split_dir+'/'+music.split('.')[0]
    if not os.path.exists(split_dir): os.mkdir(split_dir)
    if not os.path.exists(chunks_split_son_dir): os.mkdir(chunks_split_son_dir)

    for i, sub_music in enumerate(chunks_split):
        sub_music.export(chunks_split_son_dir+'/'+str(i)+'.'+format, format=format)


def ruduce_noise(music):
    '''
    Reduce the noice of a music file
    '''
    music_np, sr = librosa.load(music)
    nr_reduce = nr.reduce_noise(y=music_np, sr=sr)
    return nr_reduce, sr


class ReduceNoiseHz():
    def __init__(self, input_wav):

        self.data, self.fs = librosa.load(input_wav, sr=None, mono=True)
        self.noise_frame = 3  # 使用前三帧作为噪声估计
        self.frame_duration = 200/1000  # 200ms 帧长
        self.frame_length = np.int16(self.fs * self.frame_duration)
        self.fft = 2048  # 2048点fft

    def reduce(self):
        noise_data = self.get_noise()

        oris = librosa.stft(self.data, n_fft=self.fft)  # Short-time Fourier transform,
        mag = np.abs(oris)  # get magnitude
        angle = np.angle(oris)  # get phase

        ns = librosa.stft(noise_data, n_fft=self.fft)
        mag_noise = np.abs(ns)
        mns = np.mean(mag_noise, axis=1)  # get mean

        sa = mag - mns.reshape((mns.shape[0], 1))  # reshape for broadcast to subtract
        sa0 = sa * np.exp(1.0j * angle)  # apply phase information
        y = librosa.istft(sa0)  # back to time domain signal

        scipy.io.wavfile.write('./output.wav', self.fs, (y * 32768).astype(np.int16))  # save signed 16-bit WAV format


    def get_noise(self):
        noise_data = self.data[0:self.frame_length]
        for i in range(1, self.noise_frame):
            noise_data = noise_data + self.data[i*self.frame_length:(i+1)*self.frame_length]
        noise_data = noise_data / self.noise_frame
        return noise_data



# 

