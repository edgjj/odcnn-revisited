# Musical Onset Detection with Convolutional Neural Networks: **revisited**

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
$ pip install -r requirement.txt
```

- Prepare audio and validation sets.

```
$ python preprocess.py -i data\train_reduced\* -o data\reduced.pkl -v
```

```
$ python preprocess.py -i data\validation -o data\validation.pkl -v
```

- Train model.

```
$ python train.py -i data\reduced.pkl -o models\reduced.pt --epochs 100 -v
```

Or, if you want to perform validation during training:
```
$ python train.py -i data\reduced.pkl -iv data\validation.pkl -o models\reduced.pt --epochs 100 -v
```

Epochs parameter is optional, defaulting `100` epochs.

- Perform transient shaping with trained model. **NOTE:** No need in preprocessing for this, it's done internally.

```
$ python infer.py -m models\reduced.pt -i cool_song.wav -o cool_transient_shaped_song.wav -v
```

Or:

```
$ python infer.py -m models\reduced.pt -i cool_song.wav -v
```

- You're great!

### Training notice
- `train_reduced` in this repo is too small for training, because of limitation of uploadable file size of git, I wasn't able to upload enough size training data.

- If you want to train model with larger data, you could download audio data and corresponding notes [here](https://taikosanjiro-humenroom.net/original/).

## Flags description
### `preprocess.py`
| Flag        | Description |
| ----------- | ----------- |
| -i      | Input song(s) path for dataset creation. Allows * wildcard, so /* will be treated as multiple songs in directory |
| -o   | Resulting pickle file path |
| -v   | Perform verbose output |
| --nhop | Hop size to be used for feature creation. Default: 512.|
| --nfft | First FFT size for making magnitude spectrum on feature creation. Default: 1024. |

### `train.py`
| Flag        | Description |
| ----------- | ----------- |
| -i      | Training songs dataset path (.pkl [pickle])  |
| -iv  | Validation song path (.pkl [pickle]) |
| -o   | Resulting model path |
| -v   | Perform verbose output |
| -e (--epochs) | Number of training epochs. Default: 100.|
| --mb | Training mini batch size. Default: 128. |

### `infer.py`
| Flag        | Description |
| ----------- | ----------- |
| -m     | Path of model to be used for inference |
| -i      | Input audio path. Currently **.wav** only |
| -o   | Output audio path. When not specified, output audio is written next to input audio path. |
| -a   | Shaper attack time in milliseconds. Sets how long each transient envelope will be |
| -r   | Shaper release time in milliseconds. Sets how frequent transients are catched |
| --damp | Use shaper in a damping mode: instead of boosting transient, it'll supress them |
| --cuda | Use CUDA if possible (actually CPU is faster there) |
| --nonquad | Don't perform quadratic on probability. May introduce more mis-detections |
| --amul | Shaper attack envelope scale. |
| --mb  | Inferencing mini batches count |
| --init | Initial audio level to be set |
| --scale | Probability envelope scale. Bigger value results in a louder shapes |
| --nhop | Hop size to be used for feature creation. Default: 512.|
| --nfft | First FFT size for making magnitude spectrum on feature creation. Default: 1024. |

## References

- [IMPROVED MUSICAL ONSET DETECTION WITH CONVOLUTIONAL NEURAL NETWORKS](http://www.ofai.at/~jan.schlueter/pubs/2014_icassp.pdf)
- [SPL Transient Designer](https://spl.audio/wp-content/uploads/transient_designer_2_9946_manual.pdf)
- [Dance Dance Convolution](https://arxiv.org/pdf/1703.06891.pdf)
- [Create with AI](http://createwith.ai/paper/20170327/393)
- Tom Mullaney's Article; [Rythm Games with Neural Networks](http://tommymullaney.com/projects/rhythm-games-neural-networks)
- Woody's Article; [Musical Data Processing with CNN](https://qiita.com/woodyOutOfABase/items/01cc43fafe767d3edf62)
