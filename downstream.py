import argparse
import torchaudio
import torch
from speechtokenizer import SpeechTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.io.wavfile import write
import numpy as np
from collections import OrderedDict
import yaml
# from huggingface_hub import snapshot_download

# snapshot_download(repo_id="fnlp/SpeechTokenizer", local_dir="model_hub")


# Set up argument parser
parser = argparse.ArgumentParser(
    description="Load SpeechTokenizer model and process audio file."
)
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the model configuration file.",
    default="model_hub/speechtokenizer_hubert_avg/config.json",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="Path to the model checkpoint file.",
    default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt",
)
parser.add_argument(
    "--wavfile",
    type=str,
    required=True,
    help="file containing Path to speech files to be processed.",
)
parser.add_argument(
    "--output_file",
    type=str,
    help="Path to save the output audio file.",
    default="example_output.wav",
)

args = parser.parse_args()

# Load model from the specified checkpoint
with open(args.config_path) as fp:
    conf = yaml.load(fp, Loader=yaml.FullLoader)

model = SpeechTokenizer(conf)
checkpoint = torch.load(args.ckpt_path, map_location="cpu")

# Create a new state dictionary without 'module.' prefix
new_state_dict = OrderedDict()
for k, v in checkpoint["model"]["Speechtokenizer"].items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # Remove 'module.' prefix
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
model.eval()

# Determine the model's expected sample rate
model_sample_rate = model.sample_rate


# Function to get tokens from file and save to tokens.txt with ID
def get_tokens(file_line):
    print(file_line)
    file_id, file_path = file_line.strip().split(" ")

    wav, sr = torchaudio.load(file_path)

    # monophonic checking
    if wav.shape[0] > 1:
        wav = wav[:1,:]

    if sr != model.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

    wav = wav.unsqueeze(0)

    # Extract discrete codes from SpeechTokenizer
    with torch.no_grad():
        codes = model.encode(wav) # codes: (n_q, B, T)

    RVQ_1 = codes[:1, :, :] # Contain content info, can be considered as semantic tokens
    RVQ_supplement = codes[1:, :, :]
    # RVQ = codes[7,:,:]    

    out = torch.cat([RVQ_1, RVQ_supplement], axis=0)
    # tokens =  RVQ.view(-1).numpy()
    line = f"{file_id}\t{' '.join(map(str, out))}\n"
    return line

if __name__ == '__main__':
#file_path is path of wav.scp
#out_file is path for output token file
    file_path = args.wavfile
    out_file = args.output_file
    with open(file_path, "r") as file:
        file_lines = file.readlines()
    output_list = []
    
    with ThreadPoolExecutor() as pool:
        output_list.extend(pool.map(get_tokens, tqdm(file_lines)))
    
    with open(out_file, "w") as f:
        for line in output_list:
            f.write(line)

print("Tokens have been extracted and saved to output file")