# Civic Issue Classifier 🏙️

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.10%2B-D00000.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning model designed to identify and classify various urban civic issues from images. This project leverages transfer learning with the MobileNetV2 architecture to accurately categorize problems such as potholes, garbage, broken roads, and more, helping to streamline urban maintenance and reporting.

## Table of Contents
* [About the Project](#about-the-project)
* [Features](#features)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
  * [Training the Model](#training-the-model)
  * [Making a Prediction](#making-a-prediction)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## About the Project
This project was built to explore the application of computer vision for social good. By automatically classifying images of civic issues, we can create a foundation for systems that could help municipalities prioritize repairs, manage resources more effectively, and improve urban living conditions. The model is trained to distinguish between six distinct classes, including a "Non-Civic" category to filter out irrelevant images.

## Features
- **Multi-Class Image Classification:** Categorizes images into 6 classes: `Broken Road`, `Broken Traffic Light`, `Drainage`, `Garbage`, `Potholes`, and `Non-Civic`.
- **Transfer Learning:** Built upon the powerful, pre-trained **MobileNetV2** model for high accuracy and efficient training.
- **Data Augmentation:** Utilizes real-time data augmentation (random flips, rotations, zooms) to create a more robust and generalized model.
- **Two-Phase Training:** Employs an initial feature-extraction phase followed by a fine-tuning phase to maximize performance.
- **Modular Scripts:** Includes separate logic for training and prediction.

## Dataset
The model was trained on a custom dataset of over 50,000 images, categorized into six classes. For the model to be trained correctly, the data must be organized in the following directory structure:

```sh
my_civic_dataset
├── Broken Road
│   ├── image1.jpg
│   └── ...
├── Broken Traffic Light
│   ├── image1.jpg
│   └── ...
├── Drainage
│   ├── image1.jpg
│   └── ...
├── Garbage
│   ├── image1.jpg
│   └── ...
├── Potholes
│   ├── image1.jpg
│   └── ...
├── Non-Civic
│   ├── image1.jpg
└── ...
```
## Model Architecture
The core of this project is a transfer learning approach with MobileNetV2.
1.  **Base Model:** A pre-trained MobileNetV2 model (trained on ImageNet) is used as a feature extractor. Its top classification layer is removed.
2.  **Freezing:** The weights of the base model are initially frozen.
3.  **Custom Head:** A custom classification head is added on top of the base model. It consists of:
    -   `GlobalAveragePooling2D` to flatten the features.
    -   `Dropout` for regularization to prevent overfitting.
    -   A final `Dense` layer with a `softmax` activation function to output probabilities for the 6 classes.
4.  **Fine-Tuning:** After initial training, the top layers of the base model are unfrozen and the entire model is retrained at a very low learning rate to fine-tune the weights for this specific task.

## Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites
- Python 3.9 or higher
- `pip` package manager

### Installation
1.  **Clone the repository**
    ```sh
    git clone [https://github.com/](https://github.com/)[Your_GitHub_Username]/[Your_Repo_Name].git
    cd [Your_Repo_Name]
    ```

2.  **Create and activate a virtual environment (recommended)**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages**
    *(First, make sure you have a `requirements.txt` file. If you don't, create one by running this command in your terminal:* `pip freeze > requirements.txt`*)*
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the Model
To train the model on your own dataset, ensure your data is structured as described in the [Dataset](#dataset) section. Then, run the training script:

```sh
python train.py 
````

*(Note: If your training code is in a Jupyter Notebook, simply run the cells in order.)*

The script will perform the initial training, followed by fine-tuning, and save the final model as `civic_issue_classifier.keras`.

### Making a Prediction

Use the provided script to make a prediction on a single image.

```sh
python predict.py --image /path/to/your/test_image.jpg
```

The output will be the predicted class and the model's confidence score.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License.


## Acknowledgements

  * [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
  * [Kaggle](https://www.kaggle.com/) for the Dataset
  * [TensorFlow](https://www.tensorflow.org/)
  * [Keras](https://keras.io/)
  * [Shields.io](https://shields.io) for the badges.

<!-- end list -->
