# Handwritten Digit Recognizer

A deep learning application that recognizes handwritten digits (0–9) using a
Convolutional Neural Network (CNN) trained on the MNIST dataset, with a
web-based GUI built using Streamlit.

---

## Project Description

This system allows users to upload an image of a handwritten digit and instantly
receive a prediction from a trained CNN model. The model achieves **99%+ accuracy**
on the MNIST test set. Accuracy can be deceiving tho as real world test I did gave me a different accuracy.

The project covers the full machine learning pipeline:
- Data loading and preprocessing
- CNN model design, training, and evaluation
- A web GUI for real-world image upload and inference

---

## Technologies Used

| Category        | Technology              |
|----------------|--------------------------|
| Language        | Python 3.9+             |
| ML Framework    | TensorFlow / Keras      |
| Dataset         | MNIST                   |
| Web GUI         | Streamlit               |
| Image Processing| Pillow (PIL)            |
| Visualization   | Matplotlib              |
| Version Control | Git & GitHub            |

---

## Project Structure

```
mnist-digit-recognizer/
│
├── train_model.py        # Train CNN and save model
├── app.py                # Streamlit web application
├── model/
│   ├── mnist_cnn.h5      # Saved trained model (generated after training)
│   └── training_history.png  # Accuracy/loss plots
├── screenshots/          # GUI screenshots for README
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Instructions to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/MelvCoded/mnist-recognizer.git
cd mnist-digit-recognizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Run this once. It will download the MNIST dataset automatically, train the CNN,
and save the model to `model/mnist_cnn.h5`.

```bash
python train_model.py
```

Expected output:
```
Test Accuracy : 99.2%
Test Loss     : 0.0261
Model saved to model/mnist_cnn.h5
```

### 4. Launch the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 5. Use the App

1. Click **Browse files** to upload a `.png`, `.jpg`, or `.jpeg` image
2. The app displays your image and a 28×28 preprocessed preview
3. The predicted digit (0–9) is shown with a confidence score
4. A full breakdown of confidence scores for all digits is displayed

---

## Tips for Best Results

- Use a **white or light background** with a **dark pen or marker**
- Write the digit **large and centered** in the image
- Avoid extra marks or borders around the digit
- Good lighting and a clear photo produce better results

---

## CNN Model Architecture

```
Input (28×28×1)
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
Flatten
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (10, Softmax) → Output: digit 0–9
```

---

## Model Performance

| Metric        | Value     |
|---------------|-----------|
| Test Accuracy | ~99.2%    |
| Test Loss     | ~0.026    |
| Training Epochs | 10      |
| Batch Size    | 64        |

---

## Screenshots

    in folder screenshots
---

## Author

- **Name:** Melvin Gurung  
- **Course:** Machine Learning  
- **GitHub:** https://github.com/MelvCoded/mnist-recognizer.git
