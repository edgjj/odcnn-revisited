import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from librosa.filters import mel
from numba import jit

class Audio:
    """
    audio class which holds music data and timestamp for notes.

    Args:
        filename: file name.
        stereo: True or False; wether you have Don/Ka streo file or not. normaly True.
    Variables:


    Example:
        >>>from music_processor import *
        >>>song = Audio(filename)
        >>># to get audio data
        >>>song.data
        >>># to import .tja files:
        >>>song.import_tja(filename)
        >>># to get data converted
        >>>song.data = (song.data[:,0]+song.data[:,1])/2
        >>>fft_and_melscale(song, include_zero_cross=False)
    """

    def __init__(self, filename, stereo=True):

        self.data, self.samplerate = sf.read(filename, always_2d=True)
        if stereo is False:
            self.data = (self.data[:, 0]+self.data[:, 1])/2
        self.timestamp = []


    def plotaudio(self, start_t, stop_t):

        plt.plot(np.linspace(start_t, stop_t, stop_t-start_t), self.data[start_t:stop_t])
        plt.show()


    def save(self, filename="./savedmusic.wav", start_t=0, stop_t=None):

        if stop_t is None:
            stop_t = self.data.shape[0]
        sf.write(filename, self.data[start_t:stop_t], self.samplerate)


    def import_tja(self, filename, verbose=False, diff=False, difficulty=None):
        """imports tja file and convert it into timestamp"""
        
        now = 0.0
        bpm = 100
        measure = [4, 4]  # hyousi
        self.timestamp = []
        skipflag = False

        with open(filename, "rb") as f:
            while True:
                line = f.readline()
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')
                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0:5] == "TITLE":
                    if verbose:
                        print("importing: ", line[6:])
                elif line[0:6] == "OFFSET":
                    now = -float(line[7:-2])
                elif line[0:4] == "BPM:":
                    bpm = float(line[4:-2])
                if line[0:6] == "COURSE":
                    if difficulty and difficulty > 0:
                        skipflag = True
                        difficulty -= 1
                elif line == "#START\r\n":
                    if skipflag:
                        skipflag = False
                        continue
                    break
            
            sound = []
            while True:
                line = f.readline()
                # print(line)
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')

                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0] <= '9' and line[0] >= '0':
                    if line.find(',') != -1:
                        sound += line[0:line.find(',')]
                        beat = len(sound)
                        for i in range(beat):
                            if diff:
                                if int(sound[i]) in (1, 3, 5, 6, 7):
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, 1])
                                elif int(sound[i]) in (2, 4):
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, 2])
                            else:
                                if int(sound[i]) != 0:
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, int(sound[i])])
                        now += 60/bpm*measure[0]
                        sound = []
                    else:
                        sound += line[0:-2]
                elif line[0] == ',':
                    now += 60/bpm*measure[0]
                elif line[0:10] == '#BPMCHANGE':
                    bpm = float(line[11:-2])
                elif line[0:8] == '#MEASURE':
                    measure[0] = int(line[line.find('/')-1])
                    measure[1] = int(line[line.find('/')+1])
                elif line[0:6] == '#DELAY':
                    now += float(line[7:-2])
                elif line[0:4] == "#END":
                    if(verbose):
                        print("import complete!")
                    self.timestamp = np.array(self.timestamp)
                    break

def make_frame(data, nhop, nfft):
    """
    helping function for fftandmelscale.
    細かい時間に切り分けたものを学習データとするため，nhop(512)ずつずらしながらnfftサイズのデータを配列として返す
    """
    
    length = data.shape[0]
    framedata = np.concatenate((data, np.zeros(nfft)))  # zero padding
    return np.array([framedata[i*nhop:i*nhop+nfft] for i in range(length//nhop)])  

def fft_and_melscale(song: Audio, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    """
    fft and melscale method.
    fft: nfft = [1024, 2048, 4096]; サンプルの切り取る長さを変えながらデータからnp.arrayを抽出して高速フーリエ変換を行う．
    melscale: 周波数の次元を削減するとともに，log10の値を取っている．
    """

    feat_channels = []
    
    for nfft in nffts:

        window = signal.blackmanharris(nfft)
        filt = mel(sr=song.samplerate, n_fft=nfft, n_mels=mel_nband, fmin=mel_freqlo, fmax=mel_freqhi)
        
        # get normal frame
        frame = make_frame(song.data, nhop, nfft)
        # print(frame.shape)

        # melscaling
        processedframe = fft(window*frame)[:, :nfft//2+1]
        processedframe = np.dot(filt, np.transpose(np.abs(processedframe)**2))
        processedframe = 20*np.log10(processedframe+0.1)
        # print(processedframe.shape)

        feat_channels.append(processedframe)
    
    if include_zero_cross:
        song.zero_crossing = np.where(np.diff(np.sign(song.data)))[0]
        print(song.zero_crossing)
    
    return np.array(feat_channels)

def fft_and_melscale_mc(song: Audio, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0):
    """
    FFT and melscale method.
    Basically, a paralellized for multiple channels version.
    """

    ret = []
    num_channels = 1 + (len(song.data.shape) >= 2)

    for c in range (num_channels):
        feat_channels = []
        for nfft in nffts:

            window = signal.blackmanharris(nfft)
            filt = mel(sr=song.samplerate, n_fft=nfft, n_mels=mel_nband, fmin=mel_freqlo, fmax=mel_freqhi)
            
            # get normal frame
            frame = make_frame(song.data.T[c], nhop, nfft)
            # print(frame.shape)

            # melscaling
            processedframe = fft(window*frame)[:, :nfft//2+1]
            processedframe = np.dot(filt, np.transpose(np.abs(processedframe)**2)) # matmul of Mel filter-bank and transposed power spectrum
            processedframe = 20*np.log10(processedframe+0.1) # dB scale
            # print(processedframe.shape)

            feat_channels.append(processedframe)

        ret.append(np.array(feat_channels))

    return ret

def milden(data):
    """put smaller value(0.25) to plus minus 1 frame."""
    
    for i in range(data.shape[0]):
        
        if data[i] == 1:
            if i > 0:
                data[i-1] = 0.25
            if i < data.shape[0] - 1:
                data[i+1] = 0.25
        
        if data[i] == 0.26:
            if i > 0:
                data[i-1] = 0.1
            if i < data.shape[0] - 1:
                data[i+1] = 0.1
    
    return data

def smooth(x, window_len=11, window='hanning'):
    
    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    
    return y