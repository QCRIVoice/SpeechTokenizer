import torchaudio
import torch
from concurrent.futures import ThreadPoolExecutor
import sys
from tqdm import tqdm
import yaml
from speechtokenizer import SpeechTokenizer
from collections import OrderedDict

'''
To run this code:
1.Set config_path and ckpt_path values
2.run "python get_tokens.py wav.scp_path output_token_path"
'''

config_path = '/alt/qvoice/Speechtokenizer/SpeechTokenizer/training_config/config.yaml'
ckpt_path = '/alt/qvoice/Speechtokenizer/SpeechTokenizer/exp/Hubert_myst_21_on2gpus_bs64/checkpoint-60000steps.pkl'

with open(config_path) as fp:
    conf = yaml.load(fp, Loader=yaml.FullLoader)

model = SpeechTokenizer(conf)
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Create a new state dictionary without 'module.' prefix
new_state_dict = OrderedDict()
for k, v in checkpoint["model"]["Speechtokenizer"].items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # Remove 'module.' prefix
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
model.eval()


# Function to get tokens from file and save to tokens.txt with ID
def get_tokens(file_line):
    file_id, file_path = file_line.strip().split("\t")

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
    RVQ = codes[7,:,:]

    tokens =  RVQ.view(-1).numpy()
    line = f"{file_id}\t{' '.join(map(str, tokens))}\n"
    return line

if __name__ == '__main__':
#file_path is path of wav.scp
#out_file is path for output token file
    file_path = sys.argv[1]
    out_file = sys.argv[2]
    with open(file_path, "r") as file:
        file_lines = file.readlines()
    output_list = []
    
    with ThreadPoolExecutor() as pool:
        output_list.extend(pool.map(get_tokens, tqdm(file_lines)))
    
    with open(out_file, "w") as f:
        for line in output_list:
            f.write(line)

print("Tokens have been extracted and saved to tokens.txt")

