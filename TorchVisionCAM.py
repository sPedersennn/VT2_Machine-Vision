import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn

# Define same model architecture
class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load("TV_CNN.pth", map_location=device))
model.to(device)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                     (0.5, 0.5, 0.5))
])

# Class names
class_names = ['Machine', 'Philips', 'Plaster', 'Torx']

# Start webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image and apply transform
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

        # Get confidence
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted.item()].item()

        # Format label with confidence
        label = f"{class_names[predicted.item()]} ({confidence*100:.1f}%)"

    # Show prediction on frame
    cv2.putText(frame, f'Predicted: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Webcam Classification', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
