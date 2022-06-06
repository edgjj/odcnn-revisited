from model import *
import argparse
import os

from librosa import resample

def to_shape1D(a, shape):
    y_ = shape[0]
    y = a.shape[0]
    y_pad = (y_ - y)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),), 
                  mode = 'constant')

def single_pole(data, sample_rate, nfft, att_time=10, rel_time=10):
    true_sr = sample_rate / nfft

    coeff1 = 1.0 - np.exp2(-1.0 / max(att_time * true_sr * 0.001, 0.02))
    coeff2 = 1.0 - np.exp2(-1.0 / max(rel_time * true_sr * 0.001, 0.02))
    result = data.copy()

    z1 = 0.0

    for i in range (len(result)):
        in_ = data[i]
        f = abs(in_)

        coeff = coeff1 if f > z1 else coeff2   
        z1 = coeff * (f - z1) + z1     
     
        result[i] = z1

    return result

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Perform ODCNN inference.')
    parser.add_argument('-m', dest='model_path', help="Path of model to be used for inference.", default='./models/ka_model.pth')
    parser.add_argument('-i', dest='input_path', help="Input path.")
    parser.add_argument('-o', dest='output_path', help="Output path.")
    parser.add_argument('-a', dest='att_time', help="TS attack time.", default=120, type=float)
    parser.add_argument('-r', dest='rel_time', help="TS release time.", default=10, type=float) # may be too tight, 40+ is more musical
    parser.add_argument('-v', dest='verbose', action='store_const', const=True, help="Verbose mode.")
    parser.add_argument('-t', '--thres', dest='threshold', help="Probability threshold.", default=0.33, type=float)

    parser.add_argument('--nhop', dest='hop_size', help="Hop size used for making magnitude spectrum.", default=512, type=int)
    parser.add_argument('--nfft', dest='nfft', help="First FFT size for making magnitude spectrum.", default=1024, type=int)

    parser.add_argument('--cuda', dest='cuda', action='store_const', const=True, help="Use CUDA.")
    #parser.add_argument('--link', dest='link', action='store_const', const=True, help="Channel linking of shaper.")
    parser.add_argument('--damp', dest='damp', action='store_const', const=True, help="Perform attack suppression instead of boosting.")
    parser.add_argument('--nonquad', dest='dont_perform_quad', action='store_const', const=True, help="Don't perform quadratic noise supression.")
    parser.add_argument('--amul', dest='att_mul', help="TS attack scaling.", default=1.5, type=float) # works best with 2.5 or more
    parser.add_argument('--mb', dest='mini_batches', help="Mini batches count.", default=4096, type=int)
    parser.add_argument('--init', dest='init_lvl', help="Output level of non-shaped audio.", default=0.5, type=float)
    parser.add_argument('--scale', dest='scale', help="Attack boost strength.", default=2, type=float)
    
    args = parser.parse_args()
    
    model_path = args.model_path
    input_path = args.input_path
    output_path = args.output_path

    do_verbose = args.verbose

    hop_size = args.hop_size
    nfft = args.nfft
    nffts = [nfft * (2 ** x) for x in range (3)]
    mini_batches = args.mini_batches
    rel_time = args.rel_time
    att_time = args.att_time
    att_mul = args.att_mul
    scale = args.scale
    init_level = args.init_lvl
    probability_thres = args.threshold
    # channel_link = args.link # we should do inferencing independent of filter calculation then, so leave this for future

    perform_damping = args.damp
    can_use_cuda = torch.cuda.is_available() and args.cuda
    dont_perform_quad = args.dont_perform_quad  

    model_device = 'cuda:0' if can_use_cuda else 'cpu'

    if do_verbose:
        print(f'Initializing model on {model_device}...')

    device = torch.device(model_device)
    net = convNet()

    if do_verbose:
        print(f'Loading model weights from {model_path}...')

    net.load_state_dict(torch.load(model_path))
    net = net.to(device)

    if do_verbose:
        print(f'Loading audio from {input_path}...')

    p_audio = Audio(input_path, stereo=True)

    if do_verbose:
        print(f'Performing multi-channel feature extraction...')

    p_features = fft_and_melscale_mc(p_audio, nhop=hop_size, nffts=nffts)

    for idx in range(len(p_features)):

        if do_verbose:
            print(f'[ channel {idx} ] Performing model inference...')
        
        with torch.no_grad():
            result = net.infer(p_features[idx], device, minibatch=mini_batches)
        
        if do_verbose:
            print(f'[ channel {idx} ] Doing probability thresholding [ threshold: {probability_thres} ]...')

        result[result < probability_thres] = 0.0

        if dont_perform_quad is None:
            if do_verbose:
                print(f'[ channel {idx} ] Performing x^2 for result...')

            result = result ** 2

        if do_verbose:
            print(f'[ channel {idx} ] Doing SPL-like shaping for inferenced probability...')

        filtered_rel = single_pole(result, p_audio.samplerate, hop_size, 0, rel_time)
        filtered_att = single_pole(result, p_audio.samplerate, hop_size, att_time, rel_time) * att_mul

        filtered_diff = (filtered_rel - filtered_att).clip(min=0) # filtered_rel

        if do_verbose:
            print(f'[ channel {idx} ] Shifting inference result 5 frames forward...')

        ka_inference = np.concatenate([np.zeros(5), filtered_diff])

        if do_verbose:
            print(f'[ channel {idx} ] Upsampling shaped probability to audio sample rate...')

        upsampled_fft = resample(ka_inference, orig_sr=p_audio.samplerate / hop_size, target_sr=p_audio.samplerate)[360:]

        if do_verbose:
            print(f'[ channel {idx} ] Making sure shapes are OK, and applying shaping curve to signal...')

        scaled = scale * to_shape1D(upsampled_fft, p_audio.data.shape)
        multiplier = (init_level - scaled if perform_damping else init_level + scaled)

        if (len(p_audio.data.shape) >= 2):
            p_audio.data[:, idx] *= multiplier
        else:
            p_audio.data = p_audio.data * multiplier

    result_path = output_path
    if result_path is None:
        filename, ext = os.path.splitext(input_path)
        result_path = f'{filename}_processed{ext}'

    if do_verbose:
        print(f'Saving resulting audio to {result_path}...')

    sf.write(result_path, p_audio.data, p_audio.samplerate)