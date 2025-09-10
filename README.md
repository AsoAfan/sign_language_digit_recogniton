# Sign Language Digit Recognition 🔢✋

A deep learning project that recognizes **hand-sign digits (0–9)** from images using a fine-tuned [MobileNet](https://keras.io/api/applications/mobilenet/).  
The model is trained on the [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset).

---

## Dataset
- Source: [Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)  
- Contains **10 classes (digits 0–9)**  
- Around **218 images per class**  
- RGB, 100×100px, with hand-signs from multiple individuals  

The dataset was split into:
- **Training set** 60% of the data samples per class
- **Validation set**: 20% of the data samples per class
- **Test set**: 20% of the data samples per class

---

## Model Architecture
- Base model: **MobileNet** pretrained on ImageNet (`include_top=False`)  
- Added layers:
  - `GlobalAveragePooling2D`
  - `Dropout(0.3)` (to reduce overfitting)
  - `Dense(10, activation='softmax')`

Transfer learning was applied:
- Convolutional base **frozen** during initial training  
- Top layers trained on the sign-digit dataset  
- Optionally, last MobileNet blocks unfrozen for fine-tuning

---

## Training
- **Optimizer:** Adam (`lr=3e-4`)  
- **Loss:** Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Batch Size:** 10  
- **Epochs:** 10
- **Augmentation:** random flips, shifts, rotations, zoom (a small amount)

### Example training log

```
Epoch 1/10
125/125 - ... - accuracy: 0.9839 - loss: 0.0387 - val_accuracy: 0.9927 - val_loss: 0.0335
Epoch 5/10
125/125 - ... - accuracy: 0.9912 - loss: 0.0309 - val_accuracy: 0.9902 - val_loss: 0.0370
Epoch 10/10
125/125 - ... - accuracy: 0.9952 - loss: 0.0202 - val_accuracy: 0.9951 - val_loss: 0.0147
```

---

## Results
- **Validation Accuracy:** ~95–99%  
- **Test Accuracy:** 95%-99%
- The model generalizes well, but performance depends on lighting/hand variations.  

---

## Usage

### 1. Clone repository
```bash
git clone https://github.com/AsoAfan/sign_language_digit_recogniton.git
cd sign_language_digit_recogniton
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Roadmap

* [ ] Add live webcam inference with OpenCV
* [ ] Expand to **letters** (ASL alphabet dataset)
* [ ] Deploy model as a web app (Streamlit / FastAPI)

---



## Acknowledgements

* [Arda Mavi](https://github.com/ardamavi) for the dataset
* TensorFlow / Keras team for MobileNet implementation
* Everyone contributing to open sign language datasets

---

## Notices
- The model trained on a limited dataset and may not generalize to all hand shapes, skin tones, or lighting conditions.
- For production use, consider collecting a larger, more diverse dataset and further fine-tuning the model.
- This project is for educational purposes and not intended for medical or accessibility applications without further validation and testing.