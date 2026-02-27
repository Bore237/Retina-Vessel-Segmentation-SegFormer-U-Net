import torch
import torch.nn.functional as F
from monai.metrics import MeanIoU

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