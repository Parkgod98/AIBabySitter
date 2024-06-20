import video as v
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

import cv2
from PIL import Image
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import dlib
import glob
import kakao
import geturl as get

# 폴더 경로 설정
base_path = '/home/stratio/Desktop/deep_man/cropped_faces'

# 클래스 이름 리스트
classes = ['Happy', 'Neutral', 'Notgood', 'Sad', 'Surpring']

class_frequencies = np.array([1331, 1147, 239, 99, 300])

# 클래스 별 가중치 계산 (정규화)
total_samples = np.sum(class_frequencies)
class_weights = total_samples / (len(class_frequencies) * class_frequencies)

def weighted_cross_entropy(y_true, y_pred):
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=class_weights)
    return tf.reduce_mean(loss)

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]
# dlib의 얼굴 감지기 및 모델 경로
detector = dlib.get_frontal_face_detector()
predictor_path = '/home/stratio/Desktop/deep_man/files/shape_predictor_68_face_landmarks.dat'

# 얼굴 특징점 예측기 초기화
predictor = dlib.shape_predictor(predictor_path)

video_url = 'https://youtube.com/shorts/qXavOMpX3hE?feature=shared'

video_path = v.download_youtube_video(video_url, '/home/stratio/Desktop/deep_man/video.mp4','my_custom_video.mp4')
frames = v.extract_frames(video_path, frame_rate=30)  


model_path = '/home/stratio/Desktop/deep_man/wandb/run-20240531_151116-ca6w504v/files/model-best.h5'

model = tf.keras.models.load_model(model_path, custom_objects={'weighted_cross_entropy': weighted_cross_entropy, 'f1_score': f1_score})

good = []
arr = []
title = []

output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_height, frame_width, _ = frames[0].shape
output_video = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
flag = 0

# 프레임마다 얼굴 인식 및 감정 분류
for frame in frames:

    faces = detector(frame)
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        good = [[x1,y1,x2,y2]]
    
    if len(good) > 0:

        face = sorted(good, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face

        roi = frame[fY:fY + fH, fX:fX + fW]
        if roi.size == 0:
          continue
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        test_image = img_to_array(roi)
        
        test_image = np.expand_dims(test_image, axis=0)

        # 이미지 분류하기
        predictions = model.predict(test_image)
        print(predictions)

        # 예측 결과 출력하기
        LABELS = ['HAPPY', "NEUTRAL", "NOTGOOD", "SAD", "SURPRISING"]
        
        if (LABELS[predictions.argmax()] != "HAPPY" and flag == 0) :
            flag = 1
            file_path = os.path.join('/home/stratio/Desktop/rrresult', "frame.jpg")
            cv2.imwrite(file_path, frame)
            saved_frame_path = file_path
            msg = "아이가 행복한 상태가 아닙니다."
            kakao.Send(saved_frame_path,msg)
        emotion_text = LABELS[predictions.argmax()]
        cv2.putText(frame, emotion_text, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 프레임 표시
    output_video.write(frame)

flag = 1
output_video.release()
print("good")