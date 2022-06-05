from model import *
from music_processor import *
import argparse
import os
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform ODCNN training.')
    parser.add_argument('-i', dest='dataset_path', help="Training songs dataset path (.pkl [pickle]).", required=True)
    parser.add_argument('-o', dest='output_path', help="Output model path.", required=True)
    parser.add_argument('-e', '--epochs', dest='epochs', help="Number of training epochs.", default=100, type=int)
    parser.add_argument('--nhop', dest='hop_size', help="Hop size used for model training.", default = 512, type=int)
    parser.add_argument('--nfft', dest='nfft', help="First FFT size for feature.", default=1024, type=int)
    parser.add_argument('--mb', dest='minibatch_size', help="Mini batch size.", default=128, type=int)

    args = parser.parse_args()

    minibatch = args.minibatch_size
    epoch = args.epochs
    dataset_path = args.dataset_path
    model_path = args.model_path
    hop_size = args.hop_size
    nfft = args.nfft

    model_name, model_ext = os.path.splitext(model_path)
    log_path = f'{model_name}_log.txt'
    
    soundlen = 15 # count of input Mel-filtered frames; original dimensionality - 3x15x80 (3 channels (1024, 2048, 4096 fft size), 15 frames, 80 mel bands)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = convNet()
    net = net.to(device)

    with open(dataset_path, mode='rb') as f:
        songs = pickle.load(f)

    net.train(songs=songs, 
    minibatch=minibatch, 
    val_song=None, 
    epoch=epoch, 
    device=device, 
    soundlen=soundlen, 
    save_place=model_path, log=log_path, don_ka=1)