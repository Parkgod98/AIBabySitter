from pytube import YouTube
import os
import cv2
import numpy as np

def download_youtube_video(url, output_dir='./', filename='video.mp4'):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    output_path = os.path.join(output_dir, filename)
    stream.download(output_path=os.path.dirname(output_path), filename=os.path.basename(output_path))
    return output_path


def extract_frames(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    success = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        raise ValueError("Failed to retrieve FPS.")
    frame_interval = max(int(fps / frame_rate), 1)  # 최소값은 1
    while success:
        success, frame = cap.read()
        if success and count % frame_interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 3)
    return face


def predict_emotion(face, model):
    processed_face = preprocess_face(face)
    prediction = model.predict(processed_face)
    emotion = np.argmax(prediction)
    return emotion


def display_emotion(frame, faces, emotions):
    for (x, y, w, h), emotion in zip(faces, emotions):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame


def emotion_to_label(emotion):
    labels = ['Happy', 'Neutral', 'Notgood', 'Sad', 'Surpring']
    return labels[emotion]