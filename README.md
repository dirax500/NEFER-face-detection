# NEFER Face Detection

A Python-based face detection project built on the [NEFER dataset](https://github.com/miccunifi/NEFER) — a neuromorphic, event-based dataset for facial expression recognition presented at CVPR 2023.

---

## Overview

This project implements face detection using the NEFER dataset, which pairs RGB and event-camera video sequences of human faces annotated with emotion labels, face bounding boxes, and facial landmarks. The repository includes the project source code and pre-trained model checkpoints.

---

## Repository Structure

```
NEFER-face-detection/
├── Project/          # Main Python source code
├── checkpoints/      # Pre-trained model weights
└── README.md
```

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch (if using deep learning checkpoints)

Install dependencies:

```bash
pip install -r requirements.txt
```

> If no `requirements.txt` is present, install manually:
> ```bash
> pip install opencv-python numpy torch torchvision
> ```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/dirax500/NEFER-face-detection.git
cd NEFER-face-detection
```

### 2. Download the NEFER dataset

Follow the instructions on the [official NEFER repository](https://github.com/miccunifi/NEFER) to download and set up the dataset.

### 3. Run the project

```bash
cd Project
python main.py
```

---

## Dataset

**NEFER** (Neuromorphic Event-based Facial Expression Recognition) is a dataset composed of paired RGB and event camera sequences. Each sequence is labeled with one of the 7 universal emotions defined by Paul Ekman:

- Anger
- Disgust
- Fear
- Happiness
- Sadness
- Surprise
- Neutral

Annotations include both bounding boxes around detected faces and facial landmark coordinates.

---

## Checkpoints

Pre-trained model weights are stored in the `checkpoints/` directory and can be loaded directly for inference without retraining.

---

## Acknowledgements

- NEFER dataset by [miccunifi](https://github.com/miccunifi/NEFER) — presented at CVPR 2023
- "Neuromorphic Event-based Facial Expression Recognition" paper

---

## License

This project is for research and educational purposes. Please refer to the [NEFER dataset license](https://github.com/miccunifi/NEFER) for dataset usage terms.
