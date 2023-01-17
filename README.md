
# ASL Alphabet Classifier
This project uses the [ASL Alphabet dataset](https://www.kaggle.com/grassknoted/asl-alphabet) from Kaggle to train a model for classifying American Sign Language (ASL) alphabet gestures. The dataset consists of 87,000 labeled images of ASL alphabet gestures, with a resolution of 200x200 pixels.

## Google Colab
To run this project, you can use the provided Google Colab notebook ```asl_rev01.ipynb```. This notebook contains all the required code and instructions for training the model and evaluating its performance.

### To use the notebook, you will need to:

* Make a copy of the notebook by clicking the "Copy to Drive" button.
* Run the cells in the notebook to train the model and evaluate its performance.
* Data Preparation: 
Before training the model, you will need to download the ASL Alphabet dataset from Kaggle and place it in the data directory. The notebook includes instructions for downloading the data using the Kaggle API.

Next, you will need to preprocess the data by running the cells in the notebook that perform the following tasks:

* Convert the images to grayscale
* Resize the images to 64x64 pixels
* Split the data into training and test sets
* Training the Model: 
To train the model, run the cells in the notebook that define the model architecture and compile the model. Then, run the cell that trains the model on the ASL Alphabet dataset using TensorFlow. The notebook will also save the trained model to a file in the models directory.

## Evaluation
To evaluate the trained model on the test set, run the cells in the notebook that load the trained model from the models directory and use it to predict the ASL alphabet gestures in the test set. The notebook will then print out the model's accuracy on the test set.

## Prediction
To use the trained model to predict the ASL alphabet gesture in a new image, run the cell in the notebook that loads the trained model from the models directory and uses it to predict the ASL alphabet gesture in the image.

### Further Reading
For more information on the ASL Alphabet dataset and the techniques used in this project, see the following resources:

* [ ASL Alphabet dataset on Kaggle ](https://www.kaggle.com/grassknoted/asl-alphabet)
* [Convolutional neural networks for image classification](https://www.kaggle.com/grassknoted/asl-alphabet)
* [TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf)
