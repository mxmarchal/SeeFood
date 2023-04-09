# SeeFood

A very simple Python script that uses the [PyTorch](https://pytorch.org/) library to classify image files as either containing a hotdog or not.

Yup, like the app [SeeFood](https://www.youtube.com/watch?v=ACmydtFDTGs) from Silicon Valley.

Apple MPS is used to accelerate the neural network on macOS.

## Requirements

- Python 3.9+
- PyTorch 1.9.0+
- TorchVision 0.10.0+
- [Apple MPS](https://developer.apple.com/machine-learning/) (macOS 12.3+ / Apple Silicon)

## Usage

For training, run the following command:

```bash
$ python3.9 train.py <path/to/train_dataset> <path/to/val_dataset>
```

For inference, run the following command:

```bash
$ python3.9 predict.py <path/to/image>
```

## Dataset

The dataset used for training is the [Hotdog or Not Hotdog](https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog) dataset from Kaggle.

![Jìan-Yáng](https://static.wikia.nocookie.net/silicon-valley/images/4/49/Jian_Yang.jpg/revision/latest?cb=20210105194213)
