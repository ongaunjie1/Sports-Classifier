# Sports Classifier App (trained using PyTorch)

# Important Note
* Pre-trained model used was EfficientNet_B0 (model pre-trained on ImageNet dataset) : https://github.com/lukemelas/EfficientNet-PyTorch
* Custom model was trained using the smallest EfficientNet, model imported using torch image model (timm) : https://huggingface.co/timm
* Model Trained on 100 different sports, refer to classes.txt for the list of different sports 
* For the purpose of using Streamlit cloud, the smallest model is chosen for a faster inference speed, with less accuracy
* Accuracy of the model is **93%** 

# Link to app
* https://sports-classifier.streamlit.app/

# How to use the app?
## Step 1: Upload the image of a sport
![image](https://github.com/ongaunjie1/Sports-Classifier/assets/118142884/5a02e2bd-c90a-4fe5-ad03-c1a886a5d901)

## Step 2: Click on the classify button
![image](https://github.com/ongaunjie1/Sports-Classifier/assets/118142884/7bd33656-099d-40da-9708-713b6e20b11f)


