````markdown
# Face Mask Detection

A real-time face mask detection system built with PyTorch, OpenCV, and MediaPipe.  
This project detects faces via webcam and classifies whether a person is wearing a mask or not using a fine-tuned ResNet18 model.

---

## Features

- Real-time face detection using MediaPipe
- Mask classification with a fine-tuned ResNet18 CNN
- Webcam live feed with bounding boxes and confidence scores
- Training pipeline with data augmentation and validation
- Performance metrics: accuracy, confusion matrix, classification report

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection
````

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download or place your dataset in the `data/` folder (make sure `data/` is in `.gitignore` to avoid uploading large files).

---

## Usage

### Training the model

```bash
python train.py
```

* The model trains for 20 epochs on your dataset.
* Saved model checkpoint: `models/resnet.pth`

### Running real-time detection

```bash
python detect.py
```

* Uses your webcam to detect faces and predict mask status.
* Press `q` to quit the webcam window.

---

## Dataset

* Use a labeled dataset with two classes: `With Mask` and `Without Mask`.
* The dataset folder should have this structure:

```
data/
├── With Mask/
│   ├── image1.jpg
│   └── ...
└── Without Mask/
    ├── image1.jpg
    └── ...
```

---

## Model Architecture

* Based on a pretrained ResNet18 model with the final fully connected layer replaced for binary classification.

---

## Contributing

Feel free to open issues or submit pull requests!
Make sure to add your dataset separately and do not commit large data files.
