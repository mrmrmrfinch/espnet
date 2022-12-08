"""
Used in ASR recipe in ESPnet2 for feature extraction.
Extract features from audio files and save them in a .pt format.

Rely on torchaudio for feature extraction.

Usage:
python3 local/extract_features.py \
            --feature_type ${feature_type} \
            --downmix ${downmix} \  
            --fs ${fs} \
            --n_mels ${n_mels} \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --output_dir ${dumpdir} \
            --feature_length ${feature_length} \
            --pad_mode ${pad_mode}
"""
import os
import torch, torchaudio
import argparse
from tqdm import tqdm

def downmix(y, mode='mix'):
    if mode == 'mix': # downmix to mono
        y = torchaudio.transforms.DownmixMono()(y)
    if mode == 'first': # only use the first channel
        y = y[0].reshape(1, -1)
    if mode == 'concat': # concatenate all channels to one
        y = y.reshape(1, -1)    
    return y 
    
def trunc(y, targetLength, mode='left'):
    if mode == 'random':
        randomIdx = torch.randint(0, y.shape[1] - targetLength, (1,))
        y = y[:, randomIdx : randomIdx + targetLength]
    if mode == 'center':
        y = y[:, (y.shape[1] - targetLength) // 2 : (y.shape[1] - targetLength) // 2 + targetLength]
    if mode == 'left':
        y = y[:, :targetLength]
    if mode == 'right':
        y = y[:, -targetLength:]
    return y

def pad(y, targetLength, mode='zero'):
    if mode == 'zero':
        y = torch.nn.functional.pad(y, (0, targetLength - y.shape[1]))
    if mode == 'tile':
        y = y.repeat(1, targetLength // y.shape[1] + 1)[:, :targetLength]
    return y

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_type', type=str, default='mel', help='Feature type, raw, mel or linear')
    parser.add_argument('--downmix', type=str, default='mix', help='Downmix, mix, use first channel or concatenate all channels')
    parser.add_argument('--fs', type=int, default=16000, help='Sampling frequency')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel basis')
    parser.add_argument('--n_fft', type=int, default=512, help='Number of FFT points')
    parser.add_argument('--n_shift', type=int, default=256, help='Shift size in points')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory to load data')
    parser.add_argument('--output_dir', type=str, default='../dump', help='Directory to dump features')
    parser.add_argument('--audio_length', type=int, default=3, help='Audio length in seconds')
    parser.add_argument('--trunc_mode', type=str, default='random', help='Truncation mode, should be one of [random, center, left, right]')
    parser.add_argument('--pad_mode', type=str, default='zero', help='Padding mode, should be one of [zero, tile]')
    args = parser.parse_args()
    
    # verify arguments
    if args.feature_type not in ['raw', 'mel', 'linear']:
        print('Unsupported feature type: {}'.format(args.feature_type))
        print('Supported feature types: raw, mel, linear')
        exit(1)
    
    if args.trunc_mode not in ['random', 'center', 'left', 'right']:
        print('Unsupported truncation mode: {}'.format(args.trunc_mode))
        print('Supported truncation modes: random, center, left, right')
        exit(1)
    
    if args.pad_mode not in ['zero', 'tile']:
        print('Unsupported padding mode: {}'.format(args.pad_mode))
        print('Supported padding modes: zero, tile')
        exit(1)
        
    sample_rate = args.fs
    
    os.makedirs(args.output_dir, exist_ok=True)
    # replicate the diretcory structure of data_dir in output_dir
    directories = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    print("Generating {} features for {} set".format(args.feature_type, directories))
    for d in directories:
        # here we assume for each directory in data_dir, 
        # it contains the output from stage 1 (data_prep.py)
        # i.e. it conatins spk2utt, text, wav.scp, utt2spk
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)
        
        # read wav.scp file.
        # wav.scp should contain all the audio files to be processed.
        wavscp = open(os.path.join(args.data_dir, d, 'wav.scp'), 'r').read().split('\n')
        for wav in tqdm(wavscp, desc='Processing {} set'.format(d)):
            if wav == '': # skip empty lines
                continue
            file = wav.split(' ')[1]
            
            # read audio file
            y, sr = torchaudio.load(file)

            # start some simple pre-processing
            if sr != sample_rate: # resample if needed
                y = torchaudio.transforms.Resample(sr, sample_rate)(y)
            if y.shape[0] > 1: # convert to mono
                y = downmix(y, mode=args.downmix)   

            # truncate or pad to make all audio the same length
            if y.shape[1] > args.audio_length * sample_rate:
                y = trunc(y, targetLength=args.audio_length * sample_rate, mode=args.trunc_mode)
                
            elif y.shape[1] < args.audio_length * sample_rate:
                y = pad(y, targetLength=args.audio_length * sample_rate, mode=args.pad_mode)
                
            if args.feature_type == 'raw':
                # raw feature, save directly
                torch.save(y, os.path.join(args.output_dir, d, file.split('/')[-1].split('.')[0] + '.pt'))
            
            if args.feature_type == 'mel':
                # mel spectrogram
                mel = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=args.n_fft, win_length=args.n_fft, hop_length=args.n_shift, n_mels=args.n_mels)(y)
                torch.save(mel, os.path.join(args.output_dir, d, file.split('/')[-1].split('.')[0] + '.pt'))
                
            if args.feature_type == 'linear':
                # linear spectrogram
                linear = torchaudio.transforms.Spectrogram(n_fft=args.n_fft, win_length=args.n_fft, hop_length=args.n_shift)(y)
                torch.save(linear, os.path.join(args.output_dir, d, file.split('/')[-1].split('.')[0] + '.pt'))

if __name__ == '__main__':
    main()