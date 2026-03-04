import torch
import torch.nn.functional as F
from monai.metrics import MeanIoU
from ignite.metrics import Metric

miou_metric = MeanIoU(
    include_background=False,
    reduction="mean"
)

def compute_metrics(eval_pred):
    # Convert numpy arrays to tensors
    logits = torch.tensor(eval_pred.predictions)
    labels = torch.tensor(eval_pred.label_ids)

    # Convert logits -> binary predictions
    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).float()
    labels = labels.float()

    # Resize labels to match predictions
    if labels.shape[2:] != preds.shape[2:]:
        labels = F.interpolate(labels, size=preds.shape[2:], mode="nearest")

    # Compute mIoU
    miou = miou_metric(preds, labels)
    miou_metric.reset()

    # Reduce to a single scalar
    miou_value = miou.mean().item() 

    return {"mIoU": miou_value}


class MonaiIgniteMetric(Metric):
    def __init__(self, monai_metric):
        self.monai_metric = monai_metric
        super().__init__()

    def reset(self):
        # Remet le compteur à zéro au début de l'époque
        self.monai_metric.reset()

    def update(self, output):
        # Car c'est dictionaire
        # On récupère les données traitées par 'post_transforms'
        if isinstance(output, list):
            # On empile les Tensors pour avoir un batch propre (B, C, H, W)
            preds = torch.stack([x["pred"] for x in output])
            labels = torch.stack([x["label"] for x in output])
        else:
            preds, labels = output["pred"], output["label"]
            
        # On vérifie qu'on envoie bien du binaire
        # self.monai_metric attend (B, C, H, W)
        self.monai_metric(y_pred=preds, y=labels)

    def compute(self):
        # Calcule la moyenne finale à la fin de l'évaluation
        res = self.monai_metric.aggregate()
        return res.item() if hasattr(res, 'item') else res