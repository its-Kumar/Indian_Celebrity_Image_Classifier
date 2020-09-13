# importing Libraries
import joblib
import os
import json
import cv2
import numpy as np
import base64
from wavelate import w2d

# Globals
__class_name_to_number = None
__class_number_to_name = None
__model = None
__base = os.path.dirname(os.path.dirname(__file__))


# Load the model
def load_saved_artifacts():
    print("loading saved artifacts.... start")
    global __class_name_to_number
    global __class_number_to_name
    file_path = os.path.join(__base, "model/class_dictioary.json")
    with open(file_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {
            v: k
            for k, v in __class_name_to_number.items()
        }

    global __model
    if __model is None:
        file_path = os.path.join(__base, 'model/saved_model.pkl')
        with open(file_path, 'rb') as f:
            __model = joblib.load(f)
    print("\nloading saved artifacts...... done")


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack(
            (scalled_raw_img.reshape(32 * 32 * 3,
                                     1), scalled_img_har.reshape(32 * 32, 1)))
        len_img_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_img_array).astype(float)
        result.append({
            'class':
            class_number_to_name(__model.predict(final)[0]),
            'class_probability':
            np.round(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary':
            __class_name_to_number
        })
    return result


def class_number_to_name(class_number):
    return __class_number_to_name[class_number]


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_path = os.path.join(
        __base, 'opencv/haarcascade/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_path = os.path.join(__base, 'opencv/haarcascade/haarcascade_eye.xml')
    eye_cascade = cv2.CascadeClassifier(eye_path)
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


if __name__ == "__main__":
    load_saved_artifacts()
    print(
        classify_image(
            None,
            "test_images/freepressjournal_import_2019_04_448464640.webp"))
    print(classify_image(None, 'test_images/nkf-P92_.jpg'))
