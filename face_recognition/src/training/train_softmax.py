from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# from keras.models import load_model
import matplotlib.pyplot as plt
# from softmax import SoftMax
import numpy as np
import argparse
import pickle

# Construct the argumet parser and parse the argument
from src.detectfaces_mtcnn.Configurations import get_logger #to get all the logs
from src.training.softmax import SoftMax


class   TrainFaceRecogModel:

    def __init__(self, args):

        self.args = args
        self.logger = get_logger()
        # Load the face embeddings

        self.data = pickle.loads(open(args["embeddings"], "rb").read()) #it will load the dumped file embeddings and open the file

    def trainKerasModelForFaceRecognition(self):
        # Encode the labels
        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"]) #extracting only the names from embeddings pickle file.
        num_classes = len(np.unique(labels)) #extracting total no. of only the unique names from labels, return total no. of persons
        labels = labels.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder(categorical_features = [0]) # ohe initialization, will appy on lables.
        labels = one_hot_encoder.fit_transform(labels).toarray()

        embeddings = np.array(self.data["embeddings"])

        # Initialize Softmax training model arguments
        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]

        # Build sofmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # Create KFold
        cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
        # this is the information that we need to captuere
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

        # Train
        for train_idx, valid_idx in cv.split(embeddings):
            # dviding the data
            X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
            his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))
            print(his.history['acc'])

            #adding the information
            history['acc'] += his.history['acc']
            history['val_acc'] += his.history['val_acc']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

            self.logger.info(his.history['acc'])

        # write the face recognition model to output
        model.save(self.args['model'])
        f = open(self.args["le"], "wb")
        f.write(pickle.dumps(le))
        f.close()
