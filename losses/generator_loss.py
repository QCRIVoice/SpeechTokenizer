import torch
import torch.nn.functional as F
import torchaudio

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel_transform = torchaudio.transforms.MelScale(n_mels=num_mels, sample_rate=sample_rate, n_stft=n_fft//2+1, f_min=fmin, f_max=fmax, norm='slaney', mel_scale="htk")
        mel_basis[str(fmax)+'_'+str(y.device)] = mel_transform.fb.float().T.to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = F.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.abs(spec) + 1e-9
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def time_reconstruction_loss(x, x_hat):
    length = min(x.size(-1), x_hat.size(-1))
    return F.l1_loss(x[:, :, :length], x_hat[:, :, :length])

def frequency_reconstruction_loss(x, x_hat, **kwargs):
    x_mel = mel_spectrogram(x.squeeze(1), **kwargs)
    x_hat_mel = mel_spectrogram(x_hat.squeeze(1), **kwargs)
    length = min(x_mel.size(2), x_hat_mel.size(2))
    return F.l1_loss(x_mel[:, :, :length], x_hat_mel[:, :, :length])


def d_axis_distill_loss(feature, target_feature):
    n = min(feature.size(1), target_feature.size(1))
    distill_loss = - torch.log(torch.sigmoid(F.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=1))).mean()
    return distill_loss

def t_axis_distill_loss(feature, target_feature, lambda_sim=1):
    n = min(feature.size(1), target_feature.size(1))
    l1_loss = torch.functional.l1_loss(feature[:, :n], target_feature[:, :n], reduction='mean')
    sim_loss = - torch.log(torch.sigmoid(F.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=-1))).mean()
    distill_loss = l1_loss + lambda_sim * sim_loss
    return distill_loss 

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            if(len(rl.size()) == 4):
                loss += torch.mean(torch.abs(rl - gl[:,:,:rl.shape[2],:]))
            else:
                loss += torch.mean(torch.abs(rl - gl[:,:,:rl.shape[2]]))

    return loss * 2

def adversarial_g_loss(y_disc_gen):
    loss = 0
    for dg in y_disc_gen:
        l = torch.mean((1-dg)**2)
        loss += l

    return loss
def LIDloss(x,label):
    loss = F.cross_entropy(x,label)

    return loss

