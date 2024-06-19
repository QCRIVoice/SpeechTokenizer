import torch
import torch.nn.functional as F


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

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
    """Hinge loss"""
    loss = 0.0
    for i in range(len(y_disc_gen)):
        stft_loss = F.relu(1 - y_disc_gen[i]).mean().squeeze()
        loss += stft_loss
    return loss / len(y_disc_gen)