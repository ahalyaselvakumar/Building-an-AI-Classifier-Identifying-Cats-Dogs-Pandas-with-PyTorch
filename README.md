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

## PROGRAM
```
#import libraries
import torch
import torch.nn as nn #for building neural networks
import torch.nn.functional as func #for training neural networks
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models #for image datasets, transformations and pretrained models
from sklearn.metrics import confusion_matrix
import seaborn as sn
```

```
import numpy as np 
import pandas as pd #for data handling and analysis
import matplotlib.pyplot as plt #for plotting graphs
import os #for file path management
```
```
train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate image randomly +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # Flip half of the images horizantally 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
#same for test image but without random augmentations
test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
```
```
#Load cat/dog images using imagefolder
root = r"C:\Users\a\DEEP\workshop(cat,dog)\Data\cat-dog-pandas"
```
```
train_data = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
test_data  = datasets.ImageFolder(os.path.join(root, "test"), transform=test_transform)
```

```
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True) #shuffle for rsndomizing training samples
test_loader = DataLoader(test_data, batch_size=2, shuffle=False)
```
```
class_names = train_data.classes
```
```
print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')
```
```
#Loads VG19 model pretrained on imagenet-helps transfer learning
VGG19model = models.vgg19(pretrained=True)
```

```
for param in VGG19model.parameters():
    param.requires_grad = False
```
```
torch.manual_seed(42)
VGG19model.classifier = nn.Sequential(
    nn.Linear(25088, 1024), #converts VGG's output feauture (25088)->hidden 1024 neurons
    nn.ReLU(), #apply ReLU activation function
    nn.Dropout(0.4), #and prevent overfitting
    nn.Linear(1024, 3), #final layer outputs 3 classess (cat/dog/pandas)
    nn.LogSoftmax(dim=1) #for stable log probabilities
)
```
```
for param in VGG19model.parameters():
    print(param.numel())
```
```

criterion = nn.CrossEntropyLoss() #measures prediction error.
optimizer = torch.optim.Adam(VGG19model.classifier.parameters(), lr=0.001) #Adam optimizer updates classifier parameters
```
```
# Set time tracking
import time
start_time = time.time()

epochs = 3
max_trn_batch = 88  # As per your dataset size
max_tst_batch = 20  # As per your test dataset size

train_losses = []
test_losses = []
train_correct = []
test_correct = []
#Loop throudh epochs and batches for training.
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        if b == max_trn_batch:
            break
        b+=1

        # Model predicts-> compute loss between predicted and actual labels.
        y_pred = VGG19model(X_train)
        loss = criterion(y_pred, y_train)

        #Claculate how many images in the batch were correctly classified
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        #Backpropagation: compute gradients and update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  # Print interim results
        if b%20==0:
          acc = trn_corr.item()*100 / ((b+1)*train_loader.batch_size)
          print(f'epoch: {i+1}  batch: {b+1} loss: {loss.item():.4f} accuracy: {acc:.2f}%')


    train_losses.append(loss)
    train_correct.append(trn_corr)

    # X_test, y_test = X_test.to(device), y_test.to(device)
    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            if b == max_tst_batch:
                break

            # Apply the model
            y_val = VGG19model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

#Diplay total training time
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
```

```
#display total test accuracy
print(test_correct)
print(f'Test accuracy: {test_correct[-1].item()*100/len(test_data):.3f}%')
```

```
# Inverse normalize the images (to view the image properly)
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
```

```
image_index = 0
im = inv_normalize(test_data[image_index][0])

#display one test image
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.title("Display one test image")
plt.axis('off')
plt.show()
```

```

len(test_data)
```
```
#Predcits whether the displayed image
VGG19model.eval()
with torch.no_grad():
    new_pred = VGG19model(test_data[image_index][0].view(1,3,224,224)).argmax()

class_names[new_pred.item()]
```
```
from sklearn.metrics import confusion_matrix
import seaborn as sn
```
```
# Create a loader for the entire test set
test_load_all = DataLoader(test_data, batch_size=20, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for X_test, y_test in test_load_all:
        y_val = VGG19model(X_test)
        predicted = torch.max(y_val, 1)[1]

        # Collect results from all batches
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

# Build confusion matrix
arr = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

# Convert to DataFrame with class names
df_cm = pd.DataFrame(arr, index=class_names, columns=class_names)

# Plot heatmap
plt.figure(figsize=(7,5))
sn.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Cat vs Dog")
plt.show()
```

## OUTPUT



<img width="518" height="312" alt="image" src="https://github.com/user-attachments/assets/3261a998-02f1-43aa-883b-1ca002d635d0" />



<img width="475" height="530" alt="image" src="https://github.com/user-attachments/assets/d2606cbf-ad7d-4cb1-8ef6-597787d196a9" />



<img width="893" height="640" alt="image" src="https://github.com/user-attachments/assets/9b7194a2-4108-458a-a699-3b2853f9a8d3" />






## Result

Successfully trained and tested using VGG19 pretrained CNN model.
