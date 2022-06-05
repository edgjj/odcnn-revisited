from model import *
from music_processor import *
import argparse
import os
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform ODCNN training.')
    parser.add_argument('-i', dest='dataset_path', help="Training songs dataset path (.pkl [pickle]).", required=True)
    parser.add_argument('-iv', dest='validation_path', help="Validation song path (.pkl [pickle]).", default=None)
    parser.add_argument('-o', dest='model_path', help="Output model path.", required=True)
    parser.add_argument('-v', dest='verbose', action='store_const', const=True, help="Verbose mode.")
    parser.add_argument('-e', '--epochs', dest='epochs', help="Number of training epochs.", default=100, type=int)
    parser.add_argument('--mb', dest='minibatch_size', help="Mini batch size.", default=128, type=int)

    args = parser.parse_args()
    
    val_path = args.validation_path
    do_verbose = args.verbose
    minibatch = args.minibatch_size
    epoch = args.epochs
    dataset_path = args.dataset_path
    model_path = args.model_path

    model_name, model_ext = os.path.splitext(model_path)
    log_path = f'{model_name}_log.txt'
    
    soundlen = 15 # count of input Mel-filtered frames; original dimensionality - 3x15x80 (3 channels (1024, 2048, 4096 fft size), 15 frames, 80 mel bands)

    model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(model_device)

    if do_verbose:
        print(f'Initializing model on {model_device}...')

    net = convNet()
    net = net.to(device)

    with open(dataset_path, mode='rb') as f:
        if do_verbose:
            print(f'Loading dataset from {dataset_path}...')

        songs = pickle.load(f)
    
    val_song = None
    if val_path:
        with open(val_path, mode='rb') as f:
            if do_verbose:
                print(f'Loading validation song from {val_path}...')
            val_song = pickle.load(f)[0]

    if do_verbose:
        print(f'Performing training...')

    net.train(songs=songs, 
    minibatch=minibatch, 
    val_song=val_song, 
    epoch=epoch, 
    device=device, 
    soundlen=soundlen, 
    save_place=model_path, log=log_path, don_ka=1)