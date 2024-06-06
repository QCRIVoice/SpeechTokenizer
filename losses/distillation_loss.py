import torch
import torch.nn as nn
import torch.nn.functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class DistillLoss(nn.Module):
    def __init__(self, D, projection_dim):
        super(DistillLoss, self).__init__()
        self.D = D
        self.projection_dim = projection_dim
        # Initialize projection matrix A as a learnable parameter
        self.A = torch.nn.Linear(projection_dim, D, bias=False)

    def forward(self, Q1, S):
        """
        Args:
            Q1 (torch.Tensor): Quantized output of the RVQ first layer, shape (batch_size, timesteps, D).
            S (torch.Tensor): Semantic teacher representation, shape (batch_size, timesteps, D).
        Returns:
            torch.Tensor: The computed loss.
        """
        # Convert tensors to numpy arrays for fastdtw
        Q1_np = Q1.detach().cpu().numpy()
        S_np = S.detach().cpu().numpy()

        # Compute DTW distance and the alignment path using fastdtw
        aligned_embeddings = []
        for i in range(Q1.shape[0]):
            distance, path = fastdtw(Q1_np[i], S_np[i], dist=euclidean)
            path_Q1, path_S = zip(*path)
            Q1_aligned = Q1[i][list(path_Q1)]
            S_aligned = S[i][list(path_S)]
            aligned_embeddings.append((Q1_aligned, S_aligned))

        Q1 = torch.stack([x[0] for x in aligned_embeddings])
        S = torch.stack([x[1] for x in aligned_embeddings])

        assert Q1.shape == S.shape
        assert Q1.shape[2] == self.D

        batch_size, timesteps, _ = Q1.shape
        
        # Project Q1 using the projection matrix A
        # AQ1 = self.A(Q1)

        loss = 0.0
        for d in range(self.D):
            AQ1_d = Q1[:, :, d]
            S_d = S[:, :, d]

            # Compute cosine similarity for dimension d
            cos_sim = F.cosine_similarity(AQ1_d, S_d, dim=1)
            sigmoid_cos_sim = torch.sigmoid(cos_sim)
            loss += torch.mean(torch.log(sigmoid_cos_sim + 1e-8))

        # Take the mean and negative
        loss = -loss / self.D

        return loss
