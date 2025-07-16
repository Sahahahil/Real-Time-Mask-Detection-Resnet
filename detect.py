# detect.py

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import mediapipe as mp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 and modify final layer for 2 classes
model = resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/resnet.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Mediapipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Inference transform (must match training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

labels = ['With Mask', 'Without Mask']
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            # Add margin
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w_box + margin)
            y2 = min(frame.shape[0], y + h_box + margin)

            face_img = frame[y1:y2, x1:x2]

            # Optional: brighten the image in low-light conditions
            face_img = cv2.convertScaleAbs(face_img, alpha=1.3, beta=30)

            try:
                input_tensor = transform(face_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)

                    conf, pred = torch.max(probs, 1)
                    conf_val = conf.item()

                    if conf_val < 0.8:
                        label = "With Mask"
                    elif conf_val > 0.85:
                        label = "Without Mask"
                    else:
                        label = "Uncertain"

            except Exception as e:
                label = "Error"
                conf_val = 0.0


            # Choose color based on label
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255) if label == "Without Mask" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf_val:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
