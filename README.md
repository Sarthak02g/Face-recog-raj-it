# Face-recog-raj-it
This face detection process actually verifies the image is face image or not. Detection process actually works on MTCNN algorithm. Object Detection using MTCNN algorithm- is a python(pip) library. It is machine learning based approach used to extract faces and features from pictures or videos.
MTCNN is a way to integrate both tasks (recognition and alignment) using multi-task learning. In the first stage it uses a shallow CNN to quickly produce candidate
windows. In the second stage, it refines the proposed candidate windows through a more complex CNN and lastly, in the third stage, it uses third CNN more complex
than the others, to further refine the result and output facial landmark positions.

The face detection technology that helps locate human face in digital images and video frames. The object detection technology that deals with detecting instances of objects in digital image and videos. The proposed automated recognition system can be divided into four main modules:
1.Data Collector model -  Data is collected in the form of images taken through camera and save them into a file with a unique name.

2.Generate Face Embeddig Model - It will use the stored images and will generate the face embedding using face landmarks.

3.Train Face Recognition Model - The data collected is trained using this module. In this each image is processed. Training data is labelled.Training pictures are used to inform the computational model whether it’s correct in identifying a face or not.In this,it uses OneHotEncoding(conversion of categorical information to numerical data),LabelEncoder algorithms and also softmax classifier for training.

4.Face Predictor Model - This module is used for predicting the faces.The prediction is done by the computer vision library known as OpenCV.If the user’s data is already stored in files then system will detect the person and display his/her name.If the user is unknown then simply it writes unknown. Here,MTCNN algorithm is used to detect the faces in front of the camera.

![WhatsApp Image 2023-03-21 at 9 26 44 AM](https://user-images.githubusercontent.com/96908360/226514774-e6199dc3-d219-4bf4-8a80-279f6222c1ff.jpeg)

![WhatsApp Image 2023-03-21 at 9 24 51 AM (1)](https://user-images.githubusercontent.com/96908360/226514826-929c640f-5930-4a9d-8ca8-b2bee65ede98.jpeg)



##Installation

Step 1: Install the zip file from the github and extract it.

Step 2: Open the folder in pycharm. 

Step 3: Install requirements.txt to create a new environment by navigating to the file location and copying it path.

'''
pip install -r requirements.txt
'''

Step 4: Run the app.py in pycharm module to get the GUI applicaion working.

![steos](https://user-images.githubusercontent.com/96908360/226513428-268350a4-c32a-495b-8766-7e18c6ebe902.jpeg

#Research
 MTCNN mainly uses three cascaded networks, and adopts the idea of candidate box plus classifier to perform fast and efficient face recognition. The model is trained on a database of 50 faces we have collected, and Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measurement (SSIM), and receiver operating characteristic (ROC) curve are used to analyse MTCNN, Region-CNN (R-CNN) and Faster R-CNN.
 
 The average PSNR of this technique is 1.24 dB higher than that of R-CNN and 0.94 dB higher than that of Faster R-CNN. The average SSIM value of MTCNN is 10.3% higher than R-CNN and 8.7% higher than Faster R-CNN. The Area Under Curve (AUC) of MTCNN is 97.56%, the AUC of R-CNN is 91.24%, and the AUC of Faster R-CNN is 92.01%. MTCNN has the best comprehensive performance in face recognition. For the face images with defective features, MTCNN still has the best effect.
 
 Hence MTCNN was utilised as it was the best model for detecting faces and allowing face to be read for a particular person even while including the occulsions and unbaised to gender and biases and intrusion of objects such as specs, masks,etc
