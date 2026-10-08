# Multimodal Crop Disease Detection

A deep learning-based multimodal system for early crop disease detection using **leaf images and weather conditions**.

## Project Overview

This project focuses on early crop disease detection by combining two types of information:

* Leaf images
* Weather conditions

The system processes image and weather information separately and combines their learned features to classify the crop disease.

## Objective

The main objective of this project is to develop a multimodal crop disease detection system that considers both the visual symptoms in leaf images and weather-related information.

Compared with an image-only approach, the multimodal approach provides additional environmental information for disease classification.

## Model Architecture

The project uses a multimodal deep learning architecture consisting of two branches.

### Image Branch

* Pretrained InceptionV3
* Input image size: 224 × 224 × 3
* Extracts visual features from crop leaf images

### Weather Branch

* Processes weather-related features
* Uses dense neural network layers to learn useful weather representations

### Feature Fusion

The features extracted from the image branch and weather branch are combined using feature concatenation.

The fused features are then passed through fully connected layers and a final Softmax layer for four-class disease classification.

## Disease Classes

The model classifies the crop leaves into four classes:

| Disease/Class       | Label |
| ------------------- | ----: |
| Bacterial Leaf Spot |     0 |
| Cercospora          |     1 |
| Healthy             |     2 |
| Yellow              |     3 |

## Dataset

The dataset contains **2,817 images**.

* Training samples: 2,253
* Testing samples: 564
* Image size: 224 × 224 × 3
* Weather features: 2

The dataset metadata contains the following information:

* Image Name
* Class Name
* Date
* Location
* Temperature
* Weather
* Device

## Technologies Used

* Python
* TensorFlow
* Keras
* InceptionV3
* NumPy
* Pandas
* Google Colab
* Jupyter Notebook

## Project Workflow

```text
Leaf Image + Weather Data
          ↓
    Data Preprocessing
          ↓
   ┌───────────────┐
   │ Image Branch  │
   │  InceptionV3  │
   └───────┬───────┘
           │
           │
           ├──── Feature Fusion ────┐
           │                        │
   ┌───────▼───────┐                │
   │ Weather Branch│                │
   │ Dense Layers  │                │
   └───────────────┘                │
                                    ↓
                             Fully Connected
                                    ↓
                              Softmax Layer
                                    ↓
                         Disease Classification
```

## Project Structure

```text
multimodal-crop-disease-detection/
│
├── data/
│   └── metadata (1).csv
│
├── model/
│
├── notebook/
│   ├── alexnet (1).ipynb
│   ├── googlenet.ipynb
│   ├── multimodel.ipynb
│   ├── prediction.ipynb
│   └── training.ipynb
│
├── .gitignore
└── README.md
```

## How to Run

The main multimodal model can be run using **Google Colab**.

1. Open `notebook/multimodel.ipynb`.
2. Open the notebook in Google Colab.
3. Connect Google Drive if required by the notebook.
4. Make sure the required dataset and files are available at the paths used in the notebook.
5. Run the cells from top to bottom.
6. Use the prediction section to perform disease classification.

## Results

The multimodal model was successfully trained and tested using the crop leaf image and weather features.

The complete training and prediction workflow is available in the Jupyter notebooks included in this repository.

## Future Improvements

* Improve model accuracy using additional training data.
* Include more weather and environmental features.
* Test additional pretrained deep learning architectures.
* Deploy the trained model as a web application.
* Provide real-time crop disease prediction.

## Author

**Gopathi Naga Priyanka**

B.Tech Computer Science and Engineering
Rajiv Gandhi University of Knowledge Technologies, Ongole
