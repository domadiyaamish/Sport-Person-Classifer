import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

img =cv2.imread('./test_images/sharapova1.jpg')
print(img.shape)
print(plt.imshow(img))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray.shape
print(gray)
plt.imshow(gray,cmap='gray')
face_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
faces=face_cascade.detectMultiScale(gray,1.3,5)
print(faces)
(x,y,w,h)=faces[0]
print(x,y,w,h)
face_img= cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(face_img)

face_img= cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(face_img)

cv2.destroyAllWindows()
for (x, y, w, h) in faces:
    face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = face_img[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()

plt.imshow(roi_color, cmap='gray')

def get_cropped_images_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


original_image = cv2.imread('./test_images/sharapova1.jpg')
plt.imshow(original_image)

cropped_image = get_cropped_images_if_2_eyes('./test_images/sharapova1.jpg')
plt.imshow(cropped_image)

cropped_image_no = get_cropped_images_if_2_eyes('./test_images/sharapova2.jpg')
cropped_image_no

path_to_data ="./dataset/"
path_to_cr_data = "./dataset/cropped/"

import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

print(img_dirs)

import shutil
if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}

# for img_dir in img_dirs:
#     count = 1
#     celebrity_name = img_dir.split('/')[-1]
#     print(celebrity_name)
#
#     celebrity_file_names_dict[celebrity_name] = []
#
#     for entry in os.scandir(img_dir):
#         roi_color = get_cropped_images_if_2_eyes(entry.path)
#         if roi_color is not None:
#             cropped_folder = path_to_cr_data + celebrity_name
#             if not os.path.exists(cropped_folder):
#                 os.makedirs(cropped_folder)
#                 cropped_image_dirs.append(cropped_folder)
#                 print("Generating cropped images in folder:", cropped_folder)
#             cropped_file_name = celebrity_name + str(count) + ".png"
#             cropped_file_path = cropped_folder + "/" + cropped_file_name
#
#             cv2.imwrite(cropped_file_path, roi_color)
#             celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
#             count += 1

import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convet to float
    imArray = np.float32(imArray)
    imArray /= 255;
    # computer coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)

    return imArray_H

im_har = w2d(cropped_image,'db1',5)
plt.imshow(im_har,cmap='gray')

celebrity_file_names_dict={}
for img_dir in cropped_image_dirs:
    celebrity_name = img_dir.split('/')[-1]
    file_list=[]
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file_names_dict[celebrity_name] =file_list
celebrity_file_names_dict

class_dict ={}
count=0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name]=count
    count+=1
class_dict

X=[]
Y=[]

for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img,(32,32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        Y.append(celebrity_name)

# len(X[0])

X=np.array(X).reshape(len(X),4096).astype(float)
X.shape

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

X_train , X_test , y_train , y_test = train_test_split(X , Y ,random_state=0)
pipe =Pipeline([('scaler', StandardScaler()), ('svc',SVC(kernel = 'rbf', C =10 ))])
pipe.fit(X_train, y_train)
pipe.score(X_test,y_test)

len(y_train)

print(classification_report(y_test, pipe.predict(X_test)))

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']

        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    }
}

scores=[]
best_estimators={}
import pandas as pd
for algo,mp in model_params.items():
    pipe =make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'],cv=100,return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model':algo,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_
    })
    best_estimators[algo]=clf.best_estimator_
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df

best_estimators

best_estimators['svm'].score(X_test,y_test)

best_estimators['random_forest'].score(X_test,y_test)

best_estimators['logistic_regression'].score(X_test,y_test)

best_clf = best_estimators['logistic_regression']

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,best_clf.predict(X_test))
cm

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

import pickle
import joblib
pickle.dump(best_clf,open('pickle_model.pkl','wb'))

# with open('final_model.pkl', 'rb') as MyReadFile:
#     __model = joblib.load(MyReadFile)
#     print(__model==best_clf)
#     print(best_clf)
filename=pickle.load(open('pickle_model.pkl','rb'))

import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))

#!/usr/bin/env python3

# How to use the Python pickle module to store arbitrary python data as a file.
# Cleaned up from the demo at: https://pythontips.com/2013/08/02/what-is-pickle-in-python/

import pickle

# Create something to store
a = ['test value','test value 2','test value 3']
# And something to compare against later
b = []

# Dump a into a pickle file as bytes
with open("testPickleFile", 'wb') as f:
  pickle.dump(a, f)

# Load from the previous pickle file as bytes
with open("testPickleFile", 'rb') as f:
  b = pickle.load(f)

# Now we can compare the original data vs the loaded pickle data
print(b)    # ['test value','test value 2','test value 3']
print(a==b)

import base64
import pickle
import joblib
import json
import numpy as np

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def load_saved_artifacts():
    print("Loading saved artifacts...")

    with open("./class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    __model = pickle.load(open('./pickle_model.pkl','rb'))

    print("Loading saved artifacts...done")

    return __class_name_to_number, __class_number_to_name, __model

def get_cv2_image_from_base64_string(b64str: str) -> np.ndarray:
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_images_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img =get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray, 1.3,5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def classify_image(image_base64_data, file_path=None):
    __model = pickle.load(open('./pickle_model.pkl','rb'))
    imgs = get_cropped_images_if_2_eyes(file_path, image_base64_data)
    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1,len_image_array).astype(float)

        result.append(__model.predict(final)[0])
    return result



def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convet to float
    imArray = np.float32(imArray)
    imArray /= 255;
    # computer coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_b64_test_image_for_virat():
    with open("m_fami.txt") as f:
        return f.read()
load_saved_artifacts()

print(classify_image(get_b64_test_image_for_virat(), None))