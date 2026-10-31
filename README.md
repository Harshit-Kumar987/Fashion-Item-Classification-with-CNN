# Fashion Item Classification with CNN

## Project Overview
Fashion Item Classification is a deep learning project that uses a Convolutional Neural Network (CNN) to classify fashion products into 10 different categories using the Fashion MNIST dataset.

The project also provides a Streamlit web application where users can upload an image and view the predicted category.

## Technologies Used
- Python
- TensorFlow and Keras
- Convolutional Neural Network (CNN)
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## Dataset
The project uses the Fashion MNIST dataset, containing grayscale images of clothing items.

- Image size: 28 × 28 pixels
- Number of classes: 10
- Training images: 60,000
- Test images: 10,000

## Fashion Categories
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Model Performance
The trained CNN model achieved approximately **93% test accuracy** in the reported evaluation.

## Project Features
- Fashion image classification
- CNN-based prediction
- Image upload interface
- Prediction confidence scores
- Class probability display

## Installation and Usage

1. Clone this repository.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Place the Fashion MNIST CSV files inside the `Dataset` folder.
4. Train the model using:
   `python train_model.py`
5. Start the application:
   `streamlit run app.py`

## Note
The dataset CSV files are not included in the GitHub repository. Download Fashion MNIST separately and place the files in the Dataset folder.

## Author
Harshit Kumar