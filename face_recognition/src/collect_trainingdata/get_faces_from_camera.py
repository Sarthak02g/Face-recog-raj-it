import sys

# after image is saved we preprocess the image using face_prerocess
# from src.insightface.src.common import face_preprocess
from src.insightface.src.common import face_preprocess


sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

# using this MTCNN algorithm we are going to detect the faces
from mtcnn.mtcnn import MTCNN
# import face_preprocess
import numpy as np
import cv2
import os
from datetime import datetime

# this class is responsible for data collection through camera
class TrainingDataCollector:

    def __init__(self, args):
        self.args = args
        # Detector = mtcnn_detector
        self.detector = MTCNN()

    def collectImagesFromCamera(self):
        # for all the image processing we are going to using open cv
        # initialize video stream
        cap = cv2.VideoCapture(0)

        # Setup some useful var
        # in the start collected faces and frames will eb equal to 0,
        # then we will create a loop and go towards no. 50, means 50 images
        faces = 0
        frames = 0
        # these are the maximum number of faces that we are going to consider variable
        max_faces = int(self.args["faces"])
        # it will return 4 values
        max_bbox = np.zeros(4)

        # if the folder is not present then create the folder in the given file path.
        if not (os.path.exists(self.args["output"])):
            os.makedirs(self.args["output"])

        while faces < max_faces:
            # it will return the frame inside frame variable
            # cap.read(), will return 2 things, 1. return something or not?, 2. it will return frame
            ret, frame = cap.read()
            frames += 1

            # to initialize the unique image name
            dtString = str(datetime.now().microsecond)
            # Get all faces on current frame
            # we are calling mtcnn here, to detect faces, and passing all the frame here
            bboxes = self.detector.detect_faces(frame)

            # if face is detected
            if len(bboxes) != 0:
                # Get only the biggest face
                max_area = 0
                # loop inside detected faces
                for bboxe in bboxes:
                    # capturing the information from box
                    bbox = bboxe["box"]
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    # extracting keyoints information
                    keypoints = bboxe["keypoints"]
                    # finding area of the bounding box in detected face
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_bbox = bbox
                        landmarks = keypoints
                        max_area = area

                max_bbox = max_bbox[0:4]

                # get each of 3 frames, for less ambiguity
                if frames % 3 == 0:
                    # convert to face_preprocess.preprocess input
                    # coordinates for the landmarks of our face.
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    # reshaping and transpose
                    landmarks = landmarks.reshape((2, 5)).T
                    # here image is resized and we will detect the face.
                    nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

                    # saving the images from the camera.
                    cv2.imwrite(os.path.join(self.args["output"], "{}.jpg".format(dtString)), nimg)
                    # introducing rectangle in front of our detected face
                    cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
                    print("[INFO] {} Image Captured".format(faces + 1))
                    faces += 1
            cv2.imshow("Face detection", frame)
            # for smooth closing of the application.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
