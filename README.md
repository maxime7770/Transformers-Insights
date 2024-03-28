# Transformers-Insights

This project contains an implementation of the Transformer Encoder architecture, inspired by the "Attention Is All You Need" paper and BERT. It also aims to understand how the Encoder architecture transforms the input data under the hood by visualizing how embeddings are adapted during training. To illustrate this, the model is trained on a binary classification task using the IMDB dataset (positive and negative movie reviews).

## Installation

Create a new Python environment

```python3.9 -m venv venv```

Install the requirements

```pip install -r requirements.txt```

## Structure

    .
    ├── assets                  # Contain generated plots
    ├── src                    
        ├── main.py             # Used to train/test the model and generate plots
        ├── utils.py               
        └── model               
            ├── layers.py       # Layers used in the Transformer architecture
            └── model.py        # Combine the layers to create the Transformer model and the classification head 

## How to train the model?

Use the following, from the root directory of the project

```python3 -m src.main```

It should also save a GIF animated plot showing the evolution of the [CLS] embedding during training.

