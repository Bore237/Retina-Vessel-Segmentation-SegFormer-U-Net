from transformers import Trainer, TrainingArguments
from src.dataset import ICRDataset, HFMonaiWrapper
from monai.losses import DiceLoss
from src.model import load_segformer
from src.utils import compute_metrics
import torch.nn.functional as F
import torch.nn as nn


class SegTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dice = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        # resize labels to match logits --> out of (128*128)
        if labels.shape[2:] != logits.shape[2:]:
            labels = F.interpolate(labels, size=logits.shape[2:], mode="nearest")

        loss = self.dice(logits, labels) + self.bce(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_segformer(config_dataset, training_args, metrics=None, compute_loss=None, 
                    callbacks = None, optimizers = None, transforms=None):
    # Charger le modèle et le processor
    model, processor = load_segformer(
        config_dataset["model_name"], 
        config_dataset["num_labels"]
    )

    # Geler les couches spécifiées
    frozen_layers = config_dataset.get("frozen_layers", [])
    for name, param in model.named_parameters():
        param.requires_grad = not any(name.startswith(layer) for layer in frozen_layers)

    # Afficher le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Fraction trainable: {trainable_params/total_params:.2%}")

    # Construire le dataset
    train_builder = ICRDataset(
        image_dir=config_dataset["image_train_dir"],
        mask_dir=config_dataset["mask_train_dir"],
        channel=config_dataset["channel"],
        img_size=config_dataset["img_size"],
        apply_repeat=config_dataset["apply_repeat"],
        apply_clahe=config_dataset["apply_clahe"],
        batch_size=config_dataset["batch_size"]
    )

    val_builder = ICRDataset(
        image_dir=config_dataset["image_val_dir"],
        mask_dir=config_dataset["mask_val_dir"],
        channel=config_dataset["channel"],
        img_size=config_dataset["img_size"],
        apply_repeat=config_dataset["apply_repeat"],
        apply_clahe=config_dataset["apply_clahe"],
        batch_size=config_dataset["batch_size"]
    )

    dataset_train = train_builder.get_dataset()
    dataset_val = val_builder.get_dataset()
    hf_dataset_train = HFMonaiWrapper(dataset_train, processor, transforms=transforms)
    hf_dataset_val = HFMonaiWrapper(dataset_val, processor, transforms=None)

    # Choisir la métrique par défaut
    metric = metrics if metrics else compute_metrics

    # Créer le trainer
    if optimizers is not None:
        trainer = SegTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset_train,
            eval_dataset=hf_dataset_val,
            compute_metrics=metric,
            callbacks=callbacks,
            optimizers=optimizers
        )
    else:
        trainer = SegTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset_train,
            eval_dataset=hf_dataset_val,
            compute_metrics=metric,
            callbacks=callbacks
        )

    # Surcharger les fonctions si nécessaire
    if compute_loss:
        trainer.compute_loss = compute_loss

    # Lancer l'entraînement
    trainer.train()