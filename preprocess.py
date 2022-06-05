
from music_processor import *

from glob import glob
import pickle
import argparse

def prepare_music(songs_path, pickle_path, delete_music=True, verbose=False, difficulty=0, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    song_places = glob(songs_path)
    songs = []

    for song_place in song_places:
        
        if verbose:
            print(f'Processing [ {song_place} ]...')
        
        song = Audio(glob(song_place+"/*.ogg")[0])

        if verbose:
            print(f'[ {song_place} ] Loading TJA...')

        song.import_tja(glob(song_place+"/*.tja")[-1], difficulty=difficulty, diff=True)
        song.data = (song.data[:, 0]+song.data[:, 1])/2 # Stereo to mono conversion

        if verbose:
            print(f'[ {song_place} ] Making mel-filtered magnitude spectrograms...')
    
        song.feats = fft_and_melscale(song, nhop=nhop, nffts=nffts, mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False)
        songs.append(song)     
    
    if delete_music:
        if verbose:
            print(f'Wiping loaded music data, leaving features alone...')

        for song in songs:
            song.data = None
    
    with open(pickle_path, mode='wb') as f:
        if verbose:
            print(f'Saving pickle to [ {pickle_path} ]...')
        pickle.dump(songs, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perform dataset preprocessing for ODCNN training.')
    parser.add_argument('-i', dest='input_path', help="Input song(s) path. Allows wildcard, so /* will use multiple directories for making dataset", required=True)
    parser.add_argument('-o', dest='output_path', help="Resulting pickle path.", required=True)
    parser.add_argument('-v', dest='verbose', action='store_const', const=True, help="Verbose mode.")
    parser.add_argument('--nhop', dest='hop_size', help="Hop size to be used for feature creation.", default=512, type=int)
    parser.add_argument('--nfft', dest='nfft', help="First FFT size for making magnitude spectrum on feature creation.", default=1024, type=int)

    args = parser.parse_args()

    do_verbose = args.verbose
    input_path = args.input_path
    pickle_path = args.output_path
    hop_size = args.hop_size
    nfft = args.nfft
    nffts = [nfft * (2 ** x) for x in range (3)]
    
    print("Preparing data...")
    prepare_music(input_path, pickle_path, nhop=hop_size, nffts=nffts, verbose=do_verbose, difficulty=0)
    print("All train data processing done!")    
       