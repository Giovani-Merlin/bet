# BET - Bi-Encoder Toolkit

BET (Bi-Encoder Toolkit) is a flexible framework that provides an efficient implementation of Bi-Encoder models using PyTorch Lightning. The toolkit is designed to facilitate the training and usage of Bi-Encoder models for various applications. While the main implementation focuses on entity linking (achieving state-of-the-art results), the architecture and design of BET enable its application in a wide range of tasks that benefit from dense vector representations and efficient data retrieval.

## Features

- Efficient implementation of Bi-Encoder models using PyTorch Lightning
- Modular and flexible codebase for easy customization and extension
- Support for diverse applications leveraging dense vector representations
- Ability to train and fine-tune models with any base model and language
- Usage of SCANN for efficient similarity search and nearest neighbor retrieval

## Architecture

BET's core architecture revolves around the Bi-Encoder model. It consists of two separate encoders: one for queries and another for the input data (e.g., abstracts of Wikipedia pages). The encoders map the input text to dense vector representations, which capture semantic information and similarities between different texts.

The framework leverages the power of PyTorch Lightning to efficiently train and manage the Bi-Encoder models. PyTorch Lightning provides a high-level interface for organizing the training loop, handling distributed training, and simplifying the process of adding custom functionality to the models.

## Usage

To use BET for your application, follow these steps:

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Clone the BET repository:

   ```bash
   git clone https://github.com/giovani-merlin/BET.git
   ```


Bi-Encoder Toolkit.
