# Smart Attendance and Engagement Detection System

### Abstract

For an effective classroom environment, it becomes important to track the activities and the
state of the students taking the lecture. This becomes increasingly important owing to the
difficulties faced by students and teachers in the transition from offline to online teaching due
to the pandemic, followed by the gradual reinstating of offline teaching. This project entails
the usage of deep learning algorithms and computer vision techniques including facial
detection and recognition for real-time attendance of the students in both offline and online
settings. Moreover, engagement detection practices, such as drowsiness detection and gaze
detection to maintain the sanctity of a classroom have also been proposed

### Architecture of attendance system
The architecture of the Smart Attendance System is as follows:
1. A camera – webcam or CCTV should be installed in front of the students so that it can
capture the face of the student.
2. After the image has been captured; the captured image is transferred into the system as
an input.
3. To circumvent the variability in the brightness of the images received, the images are
converted to grayscale.
4. Futher, OpenCV’s haarcascade model is used to detect faces and then extracting these
facial images for the face recognition systems.
5. If the user is not already registered, they can at this point use the system to capture
images to be used for recognition and register themselves with our system.
6. In the next step, the LBPH algorithm has been used to recognise faces of the registered
users.
7. Finally, the attendance of the users detected i.e., their name, ID and the current date and
time are exported and saved as a CSV file.

### Drowsiness detection
1. Finding facial landmarks - For this, we use Dlib’s face landmark estimation. The library allows us to return sixty-eight
specific points (landmarks) including the upper part of the chin, the skin fringe of every eye,
the inner fringe of the eyebrow, etc.
2.  Calculating the eye aspect ratio - The above detected landmarks are then used to calculate the eye aspect ratio.
3.  Detecting drowsiness - If the calculated EAR is below a given threshold, a warning is displayed on the screen

### Running the system
Clone this repository and set the 'code' folder to be the working directory. Then run the main.py file. This will display the menu to access the different functionalities of the system.

##### Copyright (c) 2021-22 Karan Malik and Rigved Alankar

