import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names, top_n=5):
    # Get the indices of the top N classes with the highest probabilities
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_class_names = [class_names[i] for i in top_indices]
    top_probabilities = [probabilities[i] for i in top_indices]

    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Display top predictions
    axarr[1].barh(top_class_names, top_probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Top 5 Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    st.pyplot(fig)  # Display the visualization using st.pyplot()


# Define a function to read class labels from the text file
def read_class_labels(file_path):
    with open(file_path, 'r') as file:
        class_labels = [line.strip() for line in file]
    return class_labels

# Specify the path to your class labels text file
class_labels_file = 'classes.txt'

# Read the class labels from the text file
class_labels = read_class_labels(class_labels_file)

class SportsClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(SportsClassifier, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

# Create an instance of the model class
model = SportsClassifier(num_classes=100)

# Load the trained model
model.load_state_dict(torch.load('./models/efficientnet_b0_v2.pth', map_location=torch.device('cpu')))
model.to(torch.device('cpu'))
model.eval()

# Define image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create a Streamlit web app
st.title("Sports Classifier")

# Upload an image
uploaded_image = st.file_uploader("Upload a sports image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image, preprocessed_image = preprocess_image(uploaded_image, transform)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    if st.button("Classify"):
        # Make predictions
        with torch.no_grad():
            model.eval()
            output = model(preprocessed_image)

        # Get the predicted class (assuming it's a classification task)
        predicted_class = torch.argmax(output, dim=1).item()

        # Display the predicted class
        st.write(f"Predicted Class: {class_labels[predicted_class]}")

        # Visualize the predictions
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #check for gpu(cuda)
        probabilities = predict(model, preprocessed_image, device)
        visualize_predictions(image, probabilities, class_labels)
