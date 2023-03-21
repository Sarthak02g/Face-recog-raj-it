import sys

# face modell will help in generating the face embeddings.
# from insightface.deploy import face_model
# from insightface.deploy import face_model
# from src.insightface.deploy import face_model
from src.insightface.deploy import face_model

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from imutils import paths
import numpy as np
# import face_model
import pickle
import cv2
import os


class GenerateFaceEmbedding:

    def __init__(self, args):
        # initializing the variables.
        self.args = args
        # setting our image size tha we have taken from web cam.
        self.image_size = '112,112'
        # this is our main model that will ha=elp in generating the face embeddings.
        self.model = "./insightface/models/model-y1-test2/model,0"
        # setting our threshold, from official doc.
        self.threshold = 1.24
        self.det = 0

    def genFaceEmbedding(self):
        # Grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.args.dataset))

        # Initialize the faces embedder because face model will help us generate the face embeddings.
        embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)

        # Initialize our lists of extracted facial embeddings and corresponding people names
        knownEmbeddings = []
        knownNames = []

        # Initialize the total number of faces processed
        total = 0

        # Loop over the imagePaths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            # image's name, present at second last index.
            name = imagePath.split(os.path.sep)[-2]

            # load the image
            image = cv2.imread(imagePath)
            # convert face to RGB color
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1))
            # Get the face embedding vector
            face_embedding = embedding_model.get_feature(nimg)

            # add the name of the person + corresponding face
            # embedding to their respective list
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            total += 1

        print(total, " faces embedded")

        # save to output
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        # openin the file in write mode, and writing our embedding data in that file.
        f = open(self.args.embeddings, "wb")
        f.write(pickle.dumps(data))
        f.close()
