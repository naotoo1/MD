import prototorch as pt
import pytorch_lightning as pl
import torch.utils.data
import prototorch.models
from sklearn.metrics import accuracy_score
import numpy as np
import glob
import cv2
import os
from keras.applications.inception_resnet_v2 import InceptionResNetV2

# Read input images and assign labels based on folder names
print(os.listdir("malaria_images/"))

SIZE = 224  # Resize images

# Capture training data and labels into respective lists
train_images = []
train_labels = []

for directory_path in glob.glob("malaria_images/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

# Convert lists to arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Capture test/validation data and labels into respective lists

test_images = []
test_labels = []
for directory_path in glob.glob("malaria_images/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

# Convert lists to arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Encode labels from text to integers.
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
print(train_labels_encoded)
print(test_labels_encoded)


# Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One hot encode y values for neural network.
from keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Load model without classifier/fully connected layers
IR_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in IR_model.layers:
    layer.trainable = False

# Trainable parameters will be 0
IR_model.summary()

# Get the feature maps using the prior model extractor
feature_extractor = IR_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

# input for the posterior model
X_for_RF = features

# Train data set
train_ds = pt.datasets.NumpyDataset(X_for_RF, y_train)

# instantiate an object for the cross entropy learning vector quantization model
model = pt.models.probabilistic.CELVQ(hparams=dict(distribution=[1, 1]),
                                      prototypes_initializer=pt.initializers.ZerosCompInitializer(38400))

# model summary
print(model)

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

trainer = pl.Trainer(max_epochs=50, enable_model_summary=False, log_every_n_steps=10)
trainer.fit(model, train_loader)

X_test_feature = IR_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

# prediction from the celvq model
y_pred = model.predict(torch.Tensor(X_test_features))

# summary of model prototypes
print(model.prototypes)

# summary of the accuracy
print(accuracy_score(y_test, y_pred))

# save the prototype model
torch.save(model, 'celvq.pt')

# load the protype model
model2 = torch.load('celvq.pt')

