import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import numpy as np

# Define model architecture updated for 128x128 input size
class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self._to_linear = None
        self._get_conv_output((3, 128, 128))
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        self._to_linear = x.numel()
        return self._to_linear

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load("TV_CNN.pth", map_location=device))
model.to(device)
model.eval()

# Define transforms (resize to 128x128)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Class names
class_names = ['Machine', 'Philips', 'Plaster', 'Torx']

# Start webcam
cap = cv2.VideoCapture(1)

# Color mapping for classes
class_colors = {
    'Machine': (0, 255, 0),  # Green
    'Philips': (0, 0, 255),  # Red
    'Plaster': (255, 0, 0),  # Blue
    'Torx': (0, 255, 255),   # Yellow
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate bounding boxes for detected objects
    # These would be obtained via a real object detection model
    # For now, we'll just simulate them by using random coordinates
    h, w, _ = frame.shape
    box = (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))  # Simulated bounding box
    label = 'Machine'  # Simulated label

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

        # Get label with confidence
        predicted_label = class_names[predicted.item()]
        label = f"{predicted_label} ({confidence*100:.2f}%)"

    # Draw bounding box with the predicted label
    color = class_colors.get(predicted_label, (255, 255, 255))  # Default to white if not in dict
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show prediction on frame
    cv2.imshow('Webcam Classification with Bounding Boxes', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()