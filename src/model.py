from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch

def load_segformer(model_name, num_labels = 1):
    # Load pre-trained SegFormer
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        ignore_mismatched_sizes=True  # <-- this allows loading while changing num_labels
    )
    processor = SegformerImageProcessor.from_pretrained(model_name)

    processor.do_resize = False
    processor.do_rescale = False
    processor.do_normalize = True

    return model, processor