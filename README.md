# Vision Transformer Implementation in PyTorch

This repository contains a PyTorch implementation of the Vision Transformer (ViT), inspired by the seminal paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). The project builds a Vision Transformer model from scratch, processes images into patches, and trains the model on standard image datasets.

## Features
- Implements the core Vision Transformer (ViT) components:
  - Patch embeddings
  - Learnable class tokens
  - Positional embeddings
  - Multi-head self-attention (MSA)
  - Transformer encoder blocks
- Fully modular design for flexibility and experimentation.
- Customizable hyperparameters for embedding size, number of heads, layers, patch size, and more.
- Supports training and evaluation on datasets like CIFAR-10, CIFAR-100, or ImageNet.

## Repository Structure
```
.
├── VisionTransformer.py  # Main implementation of the ViT model
├── train.py              # Script to train the ViT model
├── utils.py              # Helper functions (data loading, metrics, etc.)
├── README.md             # Documentation for the repository
├── requirements.txt      # Required Python packages
└── datasets/             # Scripts for preparing datasets (optional)
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.10 or higher
- torchvision

Install the required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Training the Vision Transformer
Run the `train.py` script to train the Vision Transformer on a dataset:
```bash
python train.py --dataset CIFAR10 --img_size 224 --patch_size 16 --embed_dim 768 \
                --num_heads 8 --num_layers 12 --mlp_dim 3072 --batch_size 32 --epochs 50
```

#### Arguments:
- `--dataset`: Dataset to use (e.g., CIFAR10, CIFAR100, ImageNet).
- `--img_size`: Input image size (default: 224).
- `--patch_size`: Size of patches (default: 16).
- `--embed_dim`: Latent vector size (default: 768).
- `--num_heads`: Number of attention heads (default: 8).
- `--num_layers`: Number of Transformer encoder layers (default: 12).
- `--mlp_dim`: Hidden size of the MLP (default: 3072).
- `--batch_size`: Batch size for training (default: 32).
- `--epochs`: Number of epochs for training (default: 50).

#### Evaluating the Model
To evaluate the trained model:
```bash
python train.py --mode evaluate --dataset CIFAR10 --checkpoint path_to_checkpoint.pth
```

### Example Output
Below is an example output of training the ViT model on CIFAR-10:
```
Epoch 1/50: Loss = 2.03, Accuracy = 45.7%
Epoch 50/50: Loss = 0.21, Accuracy = 91.2%
```

## Customization
This implementation is modular, making it easy to:
1. Change the dataset by modifying `datasets/` or using custom PyTorch DataLoader scripts.
2. Experiment with different patch sizes, embedding dimensions, and Transformer configurations by updating the `VisionTransformer` parameters.
3. Extend the project to include additional tasks like object detection or segmentation.

## References
- ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Contributions
Feel free to open issues or submit pull requests to improve this repository. Contributions are always welcome!

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

