# Transformer Model for Translation

This repository contains an implementation of the Transformer model for machine translation, inspired by the "Attention is All You Need" paper by Vaswani et al. The model is built from scratch using Python and PyTorch.

## Features

- Implements the Transformer architecture with self-attention and multi-head attention mechanisms.
- Supports both encoder and decoder stacks as described in the original paper.
- Customizable hyperparameters for model size, number of layers, and attention heads.
- Trained on a sample dataset for language translation tasks.

## Requirements

- Python 3.8+

Install dependencies using:

```bash
pip install -r requirements.txt
```

## File Structure

- `transformer`: Core implementation of the Transformer model.
- `configs` : Include configurations files for training.
- `train.py`: Script to train the model on a translation dataset.

## Usage

1. Train the model:
   ```bash
   python train.py --config configs/en2it.yaml
   ```

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Acknowledgments

This implementation was inspired by tutorial videos and documentation available online.
