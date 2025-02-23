import torch
import fiftyone as fo
import fiftyone.core.labels as fol
import os
import numpy as np
from torchvision import transforms
from PIL import Image

def create_fiftyone_dataset(dataset, predicted_labels, dataset_name="cifar100-test", save_dir="data/cifar100_images"):
    """
    Creates a FiftyOne dataset from pytorch dataset

    Args:
        dataset (Dataset): PyTorch test dataset (should return (image, label) on indexing)
        predicted_labels (list or numpy array): Flat array of predicted labels (same length as dataset)
        dataset_name (str): Name of the FiftyOne dataset
        save_dir (str): Directory to save images

    Returns:
        fo.Dataset: The created FiftyOne dataset
    """
    
    assert len(predicted_labels) == len(dataset), "Mismatch between dataset size and predicted labels"

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
        print(f"Dataset '{dataset_name}' has been deleted")

    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.ToPILImage()

    samples = []

    for i in range(len(dataset)):
        img, true_label = dataset[i]
        pred_label = predicted_labels[i]

        img_pil = transform(img)
        img_path = os.path.join(save_dir, f"image_{i}.png")
        img_pil.save(img_path)

        sample = fo.Sample(
            filepath=img_path,
            ground_truth=fol.Classification(label=str(true_label)),
            predicted_label=fol.Classification(label=str(pred_label)),
        )

        samples.append(sample)

    dataset = fo.Dataset(dataset_name)
    dataset.add_samples(samples)

    print(f"Dataset '{dataset_name}' has been created with {len(samples)} samples.")

    return dataset

# Example Usage:
# dataset = create_fiftyone_dataset(test_dataset, predicted_labels)
# session = fo.launch_app(dataset)
