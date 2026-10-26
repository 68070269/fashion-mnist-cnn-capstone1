# Fashion-MNIST CNN Capstone

A small, reproducible image-classification project for the Fundamentals of Deep Learning course. The project trains a convolutional neural network (CNN) from scratch to classify Fashion-MNIST images into 10 clothing categories.

> **Status:** Project in progress. Replace the result and failure-case placeholders below with measurements and examples from the final Colab run. Do not report estimated results.

## Problem

Given a 28 × 28 grayscale image, predict one of these categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot.

## Dataset

This project uses [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), a public dataset of 70,000 labeled grayscale images. The notebook downloads the dataset when it runs; the image files are not stored in this repository.

The official test set is kept separate from training and validation:

| Split | Number of images | Use |
| --- | ---: | --- |
| Train | 54,000 | Fit model parameters |
| Validation | 6,000 | Select the best training epoch |
| Test | 10,000 | Final evaluation after model choices are fixed |

A random seed of `42` is used for the train/validation split and model setup.

## Model

The CNN is implemented in `src/model.py` and trained from scratch:

1. Convolution, 1 to 32 channels, ReLU, and 2 × 2 max pooling
2. Convolution, 32 to 64 channels, ReLU, and 2 × 2 max pooling
3. Flatten, fully connected layer with 128 units, ReLU, and dropout (`p=0.25`)
4. Fully connected output layer with 10 class logits

The model is trained with cross-entropy loss and Adam (`lr=0.001`, `weight_decay=0.0001`). The notebook currently uses batches of 128 for training, batches of 256 for evaluation, and 10 epochs. The best checkpoint is selected by validation accuracy.

## Baseline and evaluation

The baseline predicts the most common class in the training split for every image. The CNN and baseline are compared on the held-out test set using accuracy. Macro-F1 and a per-class report are also recorded to show performance across all categories.

| Model | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Majority-class baseline | _Add measured value_ | N/A |
| CNN | _Add measured value_ | _Add measured value_ |

Add a short analysis of the learning curves and at least one concrete test-set failure case here. For example, record the true label and predicted label from an image shown by the notebook. Do not use a guessed example.

## Run in Google Colab

1. Open `notebooks/fashion_mnist_cnn.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type** and choose a GPU if one is available. CPU execution is also possible but may take longer.
3. Run the notebook cells in order. The notebook downloads Fashion-MNIST, creates the train/validation split, trains the CNN, selects the best checkpoint using validation results, and evaluates the test set once at the end.
4. Save the notebook and record the final test metrics and failure analysis in this README.

The notebook and `src/model.py` should be committed to this repository. Do not commit dataset files, API keys, or other secrets. A trained checkpoint may be included as an optional demo artifact, but it does not replace the training code.

## Repository layout

```text
fashion-mnist-cnn-capstone/
├── README.md
├── requirements.txt
├── src/
│   └── model.py
├── notebooks/
│   └── fashion_mnist_cnn.ipynb
└── reports/
    └── figures/
```

## Dependencies

The notebook uses PyTorch, Torchvision, scikit-learn, and Matplotlib. Record the versions used for the final run in `requirements.txt` so the environment can be reproduced.

## References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* [Dataset repository](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch: Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
