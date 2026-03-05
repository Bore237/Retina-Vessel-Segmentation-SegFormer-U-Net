import torch
import torch.nn.functional as F

def infrerence(model, image_batch, threshold = 0.2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    with torch.no_grad():
        outputs = model(pixel_values=image_batch)
        logits = outputs.logits  # Forme: [1, num_labels, H/4, W/4]

    print("Running on:", device)

    # 3. Redimensionnement à la taille originale
    # SegFormer produit des sorties au 1/4 de la résolution
    upsampled_logits = F.interpolate(
        logits,
        size=(512, 512), # (height, width)
        mode='bilinear',
        align_corners=False
    )

    # 4. Obtenir les classes par pixel
    pred_seg = upsampled_logits.squeeze()
    proba = torch.sigmoid(pred_seg)  
    mask = (proba > threshold).float().cpu().numpy()

    return mask
