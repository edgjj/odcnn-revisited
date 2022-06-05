# Musical Onset Detection with Convolutional Neural Networks **revisited**

## Abstract
>Musical onset detection is one of the most elementary tasks in music analysis, but still only solved imperfectly for polyphonic music signals. Interpreted as a computer vision problem in spectrograms, Convolutional Neural Networks (CNNs) seem to be an ideal fit. On a dataset of about 100 minutes of music with 26k annotated onsets, we show that CNNs outperform the previous state-of-the-art while requiring less manual preprocessing. Investigating their inner workings, we find two key advantages over hand-designed methods: Using separate detectors for percussive and harmonic onsets, and combining results from many minor variations of the same scheme. The results suggest that even for well-understood signal processing tasks, machine learning can be superior to knowledge engineering.

## Description

This repository is implementation of Jan Schlüter and Sebastian Böck's ["IMPROVED MUSICAL ONSET DETECTION WITH CONVOLUTIONAL NEURAL NETWORKS"](http://www.ofai.at/~jan.schlueter/pubs/2014_icassp.pdf)

- Model architecture: simple convolutional neural network
![](./image/model.png)
- Model output: probability of onset
![](./image/result.jpg)

Original model's implementation by @seiichiinoue was good enough, but it had a major problem: non-flexibility.
It was bound to Taikosanjiro game, and it took me a while to figure what was actually happening.

This version aims to make model usage more flexible.

## Current changes
- Moved preprocessing into a single `preprocessing.py` source, allowing usage of non-standard hop/FFT sizes
- Preprocessing is now carried out through single function, instead of 4 (music_for_train, music_for_validation, etc..)
- Instead of synthesizing don/ka for inference, I do a transient shaping for input audio file
- Remove Numba JIT usage (since basically JIT was fallbacking to object mode)

## Remaining changes
- Make TJA-independent model training

## Usage
### Requirements

- **git-lfs** (for downloading large dataset)
- python3
- pytorch
- soundfile
- librosa
- tqdm

### Typical run steps

- Install requirements

```
$ pip install -r reqirement.txt
```

- Prepare audio dataset.

```
$ python preprocess.py -i data\train_reduced\* -o data\reduced.pkl -v
$ python preprocess.py -i data\test -o data\test.pkl -v
```

- Train model. (unfinished yet)

```
$ python train.py -i data\reduced.pkl -o data\reduced.pt --epochs 100
```

- then predict onset probability with trained model.

```
$ python infer.py -i cool_song.wav -o cool_transient_shaped_song.wav
```

### Training notice
- `train_reduced` in this repo is too small for training, because of limitation of uploadable file size of git, I wasn't able to upload enough size training data.

- If you want to train model with larger data, you could download audio data and corresponding notes [here](https://taikosanjiro-humenroom.net/original/).


## References

- [IMPROVED MUSICAL ONSET DETECTION WITH CONVOLUTIONAL NEURAL NETWORKS](http://www.ofai.at/~jan.schlueter/pubs/2014_icassp.pdf)
- [Dance Dance Convolution](https://arxiv.org/pdf/1703.06891.pdf)
- [Create with AI](http://createwith.ai/paper/20170327/393)
- Tom Mullaney's Article; [Rythm Games with Neural Networks](http://tommymullaney.com/projects/rhythm-games-neural-networks)
- Woody's Article; [Musical Data Processing with CNN](https://qiita.com/woodyOutOfABase/items/01cc43fafe767d3edf62)
