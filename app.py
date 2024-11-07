import streamlit as st
import matplotlib.pyplot as plt
from model import LeNet_5
import cv2
import torchvision.transforms as transforms
import torch

st.write("""
        # Simple MNIST Classifier App
        """
        )

def image_input():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = plt.imread(uploaded_file)
        return image
    return None

def to_gray_scale(image):
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image


def process_gray(grayimage):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    grayimage = transform(grayimage)
    grayimage = grayimage.unsqueeze(0)  # Add batch dimension
    return grayimage

user_input = image_input()
if user_input is not None:
    gray = to_gray_scale(user_input)
    st.image(gray, caption='Uploaded Image.', width=300)  # Change the width as needed
    gray = process_gray(gray)
else:
    gray = None
    st.write("Please upload an image file.")
    


if gray is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LeNet_5().to(device)
    model.load_state_dict(torch.load('best_lenet5.pth', map_location=device))  # Load model on the correct device
    model.eval()
    
    gray = gray.to(device)
    
    output = model(gray)
    _, pred = torch.max(output, 1)
    st.write(f"Prediction: {pred.item()}")
    st.write(f"Confidence: {torch.nn.functional.softmax(output, dim=1)[0][pred].item() * 100:.2f}%")


st.write("""
         ### Why this model sucks?
            - It's a one of the oldest CNN model
            - This model was trained on the MNIST dataset only, so it might not perform well on real-world images
            - The model might not even capture the features of the image
            - If you are trying for a better performance, you can check out ResNet, EfficientNet, ViT etc.
            """
        )