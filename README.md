# Face Recognition Attendance System (VGG-Face)

A webcam attendance system that recognises faces with the pretrained VGG-Face model, with no training needed: faces are compared as feature vectors.

> 📦 **Status:** Archived · Dec 2023 · early prototype for the [Siamese network attendance system](https://github.com/yx-elite/face-recognition-attendance-siamese), EPE461 Industrial Machine Vision, Universiti Sains Malaysia

## Overview

Training a face recognition model from scratch needs a lot of data and compute. This project takes a shortcut: it uses **VGG-Face**, a model already trained on millions of face images, as a **feature extractor**.

Each face is turned into a vector that describes its features. Two photos of the same person produce vectors that are close together, and photos of different people produce vectors far apart. So recognising someone only means comparing vectors, without training anything for new people.

## Architecture

<p align="center">
  <img src="docs/vggface-flow.svg" alt="VGG-Face verification flow: capture a face, compare its VGG-Face vector with 5 reference images by cosine and Euclidean distance, and log attendance if at least 4 of 5 match" width="100%">
</p>

**1. Registration** (`face_register_manual_csv.py`)
- Registers a person into a CSV user database and captures reference face images from a webcam
- Faces are detected with OpenCV's Haar cascade, then cropped and resized

**2. Feature extraction**
- VGG-Face (a 16-layer VGG network) loaded with pretrained weights
- The final classification layer is removed, so the network outputs a **2,622-dimension face descriptor** instead of a class label

**3. Verification** (`face_recognition_attendance.py`)
- The captured face is compared with **5 of the person's reference images**
- Each pair counts as a match only if **both** distance checks pass:
  - **Cosine distance** < 0.23 (are the vectors pointing the same way?)
  - **Euclidean distance** < 90 (are the vectors close together?)
- Attendance is recorded only if **at least 4 of the 5 comparisons match**, and is saved to a daily CSV log

## Experiments

The `archive/` folder shows the approaches I tried before settling on this design:

- **VGG16 transfer learning as a classifier** (`train_model_vgg16*.py`): froze the pretrained ImageNet layers and trained a new softmax head, with data augmentation and early stopping. The catch is that a classifier has to be **retrained every time someone new joins**, which doesn't suit an attendance system.
- **Face detection: Haar cascade vs. MTCNN** (`face_register_haar*.py`, `face_register_mtcnn.py`): compared OpenCV's lightweight Haar cascade with the deep-learning MTCNN detector. The final app uses Haar cascade.

This led to the switch from classification to **verification**: compare faces instead of classifying them.

## Example

A verification between two photos of the same person (`vggface_model_train.ipynb`):

| Check | Result | Threshold | Pass |
|---|---|---|---|
| Cosine distance | 0.165 | < 0.23 | ✅ |
| Euclidean distance | 51.5 | < 90 | ✅ |

The thresholds were tuned by hand from test comparisons like this one.

## Tech Stack

Python · Keras / TensorFlow · OpenCV · NumPy · VGG-Face pretrained weights

## Getting Started

```bash
pip install -r requirements.txt
python face_register_manual_csv.py    # register a person and capture reference images
python face_recognition_attendance.py # verify a face and record attendance
```

Download the pretrained weights ([vgg_face_weights.h5](https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5)) into `model/`. Face data and notebook outputs aren't included, to protect the privacy of the people in the dataset. Paths are Windows-style, and the webcam is set to device index `1`.

## Lessons Learned

- **Pretrained models are a strong starting point:** VGG-Face recognised faces well with no training at all
- **Two metrics are safer than one:** requiring both cosine and Euclidean distance to agree cut down false matches
- **Hand-tuned thresholds don't travel well:** values that worked with one camera and lighting setup needed retuning elsewhere
- **Why we moved on:** VGG-Face is pretrained, so the network itself was a black box. For the final system we designed and trained our own CNN as a Siamese network, to understand and control every layer
- **Verification fits attendance better than classification:** new people shouldn't require retraining

## Credits

- **VGG-Face model:** O. M. Parkhi, A. Vedaldi and A. Zisserman, "[Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/)", British Machine Vision Conference, 2015
- **Keras weights and approach:** Sefik Ilkin Serengil's [VGG-Face tutorial](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and [deepface](https://github.com/serengil/deepface) weights

## Related Work

This is one of three approaches I explored for face recognition:

- [face-recognition-attendance-siamese](https://github.com/yx-elite/face-recognition-attendance-siamese): the next step, a Siamese network trained to learn face similarity directly (99.44% validation accuracy)
- [face-recognition-cnn-vs-transfer-learning](https://github.com/yx-elite/face-recognition-cnn-vs-transfer-learning): custom CNN vs. ResNet50 transfer learning in MATLAB, with an ablation study
