# Data Organization

This repository contains image filenames for both pre-training and fine-tuning phases. The original images can be accessed through the Sewer-ML dataset [[citation](https://vap.aau.dk/sewer-ml/)].

## Pre-training Data

The pre-training data is organized into three incrementally larger sets by combining subfolders in a fixed order:

- **Batch 1**: 104,013 unique images (train03)
- **Batch 2**: 208,026 images (train03 + train04)
- **Batch 3**: 312,039 images (train03 + train04 + train05)

Each batch contains image filenames in txt format. To access the actual images, please refer to the original Sewer-ML dataset.

## Fine-tuning Data

The fine-tuning data is structured into two main directories:

```
fine-tuning/
├── images/          # Not included - Available through Sewer-ML dataset
└── annotations/     # Included in this repository
    ├── train/      # Contains 5 batches
    ├── val/        # Contains 5 batches
    └── test/       # Original Sewer-ML validation set
```

### Train and Validation Batch Organization

We created 5 batches with increasing numbers of labeled images per defect class:

| Batch   | Images per Class (Train-Val) | Non-defective Images | Total Images |
|---------|----------------------------|---------------------|--------------|
| Batch 1 | 50-15                      | 1,105              | 2,210        |
| Batch 2 | 100-30                     | 2,210              | 4,420        |
| Batch 3 | 200-60                     | 4,420              | 8,840        |
| Batch 4 | 400-120                    | 8,840              | 17,680       |
| Batch 5 | 800-240                    | 17,680             | 35,360       |

### Test Set
For evaluation, we use the original validation set from the Sewer-ML dataset to ensure consistent comparison with other methods in the literature. This test set remains constant across all experiments, regardless of the training batch size.

### Data Distribution
- Each training/validation batch maintains a balanced distribution between defective and non-defective (ND) images
- The number of ND images equals the total number of defective images in each batch
- Training and validation splits are provided for each batch

## Accessing the Data

1. First, obtain access to the Sewer-ML dataset through [link to Sewer-ML paper/repository]
2. Clone this repository to get the image filenames and annotations
3. Use the provided txt files to locate the corresponding images in your Sewer-ML dataset

## Citation

If you use this dataset, please cite both:

1. The original Sewer-ML dataset:
```bibtex
@InProceedings{Haurum_2021_CVPR,
author = {Haurum, Joakim Bruslund and Moeslund, Thomas B.},
title = {Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021},
pages = {13456-13467}
}
```

2. This work:
```bibtex
Yildizi, T., Jia, T., Langeveld, J., & Taormina, R. (2026). Self-Supervised Learning Approach for Multi-label Sewer Defect Classification. Automation in Construction, ..
```
