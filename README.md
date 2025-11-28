# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch
## Aim

To build an image classifier that can accurately identify whether a given image is of a cat or a dog using the VGG19 deep learning model.

## Algorithm

#### Import Libraries:
Load all necessary libraries such as PyTorch, Torchvision, NumPy, Matplotlib, Seaborn, and Scikit-learn.

torch and torch.nn are used for neural network building and training.

torchvision.datasets and transforms handle image loading and preprocessing.

matplotlib and seaborn help in visualization.

confusion_matrix from sklearn helps to evaluate model performance.

#### Image Preprocessing (Transforms):
Apply a series of transformations to prepare images for the VGG19 model.

RandomRotation(10): Slightly rotates images to make the model rotation-invariant.

RandomHorizontalFlip(): Flips images horizontally to generalize better.

Resize(224) and CenterCrop(224): Ensures input images are resized and cropped to 224×224 pixels.

ToTensor(): Converts image data into tensor form.

Normalize(): Scales pixel values using ImageNet mean and standard deviation to match pretrained model expectations.

#### Dataset Loading:
Use ImageFolder to load images from folders named “cat” and “dog”.

The dataset is divided into training and testing folders.

Each image is automatically labeled based on its folder name.

#### DataLoader Creation:
The dataset is converted into batches using DataLoader for efficient training.

Batch size controls how many images are processed at once.

Shuffling ensures the model sees images in random order.

#### Model Selection (VGG19):

Load the VGG19 pretrained model from PyTorch’s model library.

Freeze its convolutional layers so only the new classifier layers are trained.

#### Modify Classifier Layer:

Replace the final layers with a new sequence of layers:

Linear → ReLU → Dropout → Linear → LogSoftmax

These layers are responsible for classifying between 2 classes: Cat and Dog.

#### Loss Function and Optimizer:

Loss Function: CrossEntropyLoss measures prediction error.

Optimizer: Adam optimizer updates model parameters to minimize loss.

#### Model Training:

For each epoch, images are passed through the model to generate predictions.

The difference between predicted and actual labels is computed as loss.

Gradients are backpropagated, and model weights are updated to reduce loss.

#### Model Evaluation:

The trained model is tested using unseen images.

Predicted results are compared against actual labels to measure accuracy.

#### Confusion Matrix Visualization:

The confusion matrix shows the number of correctly and incorrectly predicted images for each class.

Seaborn heatmap is used for better visualization.

## Code Explanation

Library Import:
Imports all required modules for model creation, image preprocessing, and visualization.

Transforms:
Each transformation step improves model accuracy and prevents overfitting.

Dataset & DataLoader:
Efficiently loads and batches the images for training and testing.

Model Initialization:
Loads pretrained VGG19, freezes feature extractor layers, and customizes the classifier for 2-class prediction.

Training Phase:
Performs forward propagation, calculates loss, performs backpropagation, and updates parameters.

Testing Phase:
Evaluates model accuracy using test images without updating gradients.

Confusion Matrix:
Visualizes classification performance, showing correctly and wrongly predicted samples.

## Result

Successfully trained and tested using VGG19 pretrained CNN model.
