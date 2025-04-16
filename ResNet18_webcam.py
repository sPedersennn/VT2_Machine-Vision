# ----- LIVE WEBCAM INFERENCE -----
model.eval()
model = model.to("cpu")  # Use CPU for webcam inference
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
class_names = train_dataset.classes

# Webcam Setup
cap = cv2.VideoCapture(0)
transform_webcam = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform_webcam(img_rgb).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probs, 0)
        label = class_names[predicted.item()]

    cv2.putText(img, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Classification", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()