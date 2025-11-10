import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.7,           # for d = 1 - cos in [0,2]
        alpha: float = 0.3,            # weight on contrastive term; (1-alpha) on MSE(sim01, label)
        pos_weight: float = 1.0,       # weight for positive pairs in contrastive term
        neg_weight: float = 1.0        # weight for negative pairs in contrastive term
    ):
        super().__init__()
        assert 0.0 <= margin <= 2.0, "margin must be in [0, 2] for d = 1 - cos"
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.margin = margin
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        output1, output2: (B, D) embeddings
        label: soft similarity target in [0,1] (works with 0/1 too)
        """
        label = label.float().view(-1)

        cos = F.cosine_similarity(output1, output2, dim=1)
        cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
        d = 1.0 - cos                      
        sim01 = (cos + 1.0) / 2.0

        y_bin = (label > 0.5).float()
        pos_term = (d ** 2)
        neg_term = F.relu(self.margin - d) ** 2

        contrastive = self.pos_weight * y_bin * pos_term + self.neg_weight * (1.0 - y_bin) * neg_term
        contrastive = contrastive.mean()

        reg = F.mse_loss(sim01, label)

        # Hybrid objective
        loss = self.alpha * contrastive + (1.0 - self.alpha) * reg
        return loss