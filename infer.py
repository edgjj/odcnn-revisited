from model import *
import pickle
import argparse

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
    parser.add_argument('-m', dest='model_path', help="Model to inference path", default='./models/ka_model.pth')
    parser.add_argument('-i', dest='input_path', help="Input path.")
    parser.add_argument('-o', dest='output_path', help="Output path.")
    parser.add_argument('-a', dest='att_time', help="TS attack time.", default=120, type=float)
    parser.add_argument('-r', dest='rel_time', help="TS release time.", default=10, type=float)
    parser.add_argument('--nhop', dest='hop_size', help="Hop size used for model training.", default=512, type=int)
    parser.add_argument('--nfft', dest='nfft', help="First FFT size for feature.", default=1024, type=int)
    parser.add_argument('--cuda', dest='cuda', action='store_const', const=True, help="Use CUDA.")
    parser.add_argument('--damp', dest='damp', action='store_const', const=False, help="Perform attack suppression instead of boosting.")
    parser.add_argument('--nonquad', dest='dont_perform_quad', action='store_const', const=True, help="Don't perform quadratic noise supression.")
    parser.add_argument('--amul', dest='att_mul', help="TS attack scaling.", default=1.5, type=float)
    parser.add_argument('--mbatch', dest='mini_batches', help="Mini batches count.", default=4096, type=int)
    parser.add_argument('--init', dest='init_lvl', help="Output level of non-shaped audio.", default=0.5, type=float)
    parser.add_argument('--scale', dest='scale', help="Attack boost strength.", default=2, type=float)
    
    args = parser.parse_args()
    
    model_path = args.model_path
    input_path = args.input_path
    output_path = args.output_path

    hop_size = args.hop_size
    nfft = args.nfft
    nffts = [nfft * (2 ** x) for x in range (3)]
    mini_batches = args.mini_batches
    rel_time = args.rel_time
    att_time = args.att_time
    att_mul = args.att_mul
    scale = args.scale
    init_level = args.init_lvl

    perform_damping = args.damp
    use_cuda = args.cuda
    dont_perform_quad = args.dont_perform_quad  

    device = torch.device('cuda:0' if (torch.cuda.is_available() and use_cuda) else 'cpu')
    net = convNet()
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)

    p_audio = Audio(input_path, stereo=True)
    p_features = fft_and_melscale_mc(p_audio, nhop=hop_size, nffts=nffts)
    
    for idx in range(len(p_features)):
        with torch.no_grad():
            result = net.infer(p_features[idx], device, minibatch=mini_batches)

        if dont_perform_quad is None:
            result = result ** 2

        filtered_rel = single_pole(result, p_audio.samplerate, hop_size, 0, rel_time)
        filtered_att = single_pole(result, p_audio.samplerate, hop_size, att_time, rel_time) * att_mul

        filtered_diff = (filtered_rel - filtered_att).clip(min=0) # filtered_rel

        ka_inference = np.concatenate([np.zeros(5), filtered_diff])

        upsampled_fft = resample(ka_inference, orig_sr=p_audio.samplerate / hop_size, target_sr=p_audio.samplerate)[250:]
 
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

    sf.write(result_path, p_audio.data, p_audio.samplerate)