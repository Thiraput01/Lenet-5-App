# Handwritten Digit Recognition using CNNs
https://lenet-5-app-thiraputball.streamlit.app/

This project implements **Part 2** of the paper *"Gradient-Based Learning Applied to Document Recognition"*, which focuses on using **Convolutional Neural Networks (CNNs)** for handwritten digit recognition with the **MNIST** dataset.

---

## Abstract

This paper explores methods for handwritten character recognition, emphasizing the power of **Convolutional Neural Networks (CNNs)** in classifying high-dimensional patterns with minimal preprocessing. It introduces **Graph Transformer Networks (GTNs)** for globally training multimodule systems and demonstrates their effectiveness in real-world applications such as **online handwriting recognition** and **bank check reading**.



---

## Part 2: MNIST CNN Implementation


![image](https://github.com/user-attachments/assets/fcb54cc9-24cc-4853-aea7-be72d4c585b5)
In this project, we focus on implementing **Part 2** of the paper, which describes the use of CNNs for recognizing digits in the **MNIST dataset**. The goal is to classify digits (0-9) with a high accuracy by leveraging the power of convolutional layers for feature extraction and pooling for dimensionality reduction.

### Key Features:
- **Convolutional Neural Network (CNN)** architecture
- Application on the **MNIST dataset** for handwritten digit classification
- Minimal preprocessing required, focusing on raw pixel data

### Steps:
1. **Load and Preprocess Data**: The MNIST dataset is preprocessed and normalized to be used for training the CNN.
2. **CNN Architecture**: The model consists of convolutional layers, activation functions, and pooling layers to extract relevant features from the images.
3. **Training**: The model is trained using gradient-based learning techniques.
4. **Evaluation**: The trained model is evaluated on test data to assess its performance.

---

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone <repository-url>
cd <project-directory>
pip install -r requirements.txt
