import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import os

video_path = r"C:\Users\affan\OneDrive\Desktop\videos commitee\drhassan.mp4"
INPUT_PATHS = [video_path]

SEQ_LENGTH = 30
PREDICTION_HISTORY_LENGTH = 15
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv']
FRAME_SIZE = (800, 1080)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, label encoder using absolute paths
model = joblib.load(os.path.join(BASE_DIR, 'mlp_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))


# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# def extract_pose_landmarks(frame):
#     results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     if results.pose_landmarks:
#         landmarks = []
#         for lm in results.pose_landmarks.landmark:
#             landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
#         return landmarks, results.pose_landmarks
#     return None, None
def extract_pose_landmarks(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return landmarks, results.pose_landmarks
    return None, None

def get_stable_prediction(history):
    if not history:
        return "Collecting..."
    most_common = Counter(history).most_common(1)[0][0]
    return most_common

# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames_with_data = []
#     pose_buffer = deque(maxlen=SEQ_LENGTH)
#     prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.resize(frame, FRAME_SIZE)
#         landmarks, pose_landmarks = extract_pose_landmarks(frame)
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_with_data = []
    pose_buffer = deque(maxlen=SEQ_LENGTH)
    prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, FRAME_SIZE)
            landmarks, pose_landmarks = extract_pose_landmarks(frame, pose)

            if landmarks:
                pose_buffer.append(landmarks)

            if len(pose_buffer) == SEQ_LENGTH:
                sequence = np.array(pose_buffer).flatten().reshape(1, -1)
                sequence_scaled = scaler.transform(sequence)
                pred = model.predict(sequence_scaled)
                pred_label = label_encoder.inverse_transform(pred)[0]
                prediction_history.append(pred_label)

            stable_prediction = get_stable_prediction(prediction_history)

            # Parse prediction like "Ahsan (Male) (Confident)"
            try:
                name, gender, gait = stable_prediction.split(" (")
                gender = gender.rstrip(")")
                gait = gait.rstrip(")")
            except:
                name = stable_prediction
                gender = "Unknown"
                gait = "Unknown"

            if pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw parsed labels
            cv2.putText(frame, f'Name: {name}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Gender: {gender}', (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'GaitStyle: {gait}', (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            frames_with_data.append(frame.copy())

    cap.release()
    return frames_with_data


def display_processed_videos(processed_videos):
    num_videos = len(processed_videos)
    frame_idx = 0
    max_len = max(len(v) for v in processed_videos)

    while frame_idx < max_len:
        for i in range(num_videos):
            if frame_idx < len(processed_videos[i]):
                frame = processed_videos[i][frame_idx]
            else:
                frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            cv2.imshow(f"Video {i+1}", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
            break
        frame_idx += 1

    cv2.destroyAllWindows()

def main():
    for path in INPUT_PATHS:
        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            print(f"❌ Unsupported file extension for: {path}")
            return

    print("🔄 Processing videos...")
    processed_videos = [process_video(path) for path in INPUT_PATHS]
    print("✅ Processing complete. Playing videos...")
    display_processed_videos(processed_videos)

    # Funtion for gait api

#
# def analyze_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     pose_buffer = deque(maxlen=SEQ_LENGTH)
#     print("Inside Recognition Process :)")
#     # No limit on prediction history to capture all predictions
#     prediction_history = []
#
#     frame_count = 0
#     valid_pose_count = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.resize(frame, FRAME_SIZE)
#         landmarks, pose_landmarks = extract_pose_landmarks(frame)
#         frame_count += 1
#
#         if landmarks:
#             valid_pose_count += 1
#             pose_buffer.append(landmarks)
#
#             if len(pose_buffer) >= 20:
#                 padded_buffer = list(pose_buffer)
#
#                 # Pad with zeros if fewer than 30 frames
#                 while len(padded_buffer) < SEQ_LENGTH:
#                     padded_buffer.append([0.0] * 132)
#
#                 sequence = np.array(padded_buffer).flatten().reshape(1, -1)
#
#                 try:
#                     sequence_scaled = scaler.transform(sequence)
#                     pred = model.predict(sequence_scaled)
#                     pred_label = label_encoder.inverse_transform(pred)[0]
#                     prediction_history.append(pred_label)
#                 except Exception as e:
#                     print("Prediction error:", e)
#                     continue
#
#     cap.release()
#     #print(f"Total frames: {frame_count}, Valid poses: {valid_pose_count}")
#     #print("Prediction history:", prediction_history)
#
#     # --------- Improved Voting Logic ---------
#     counter = Counter(prediction_history)
#     if not counter:
#         final_label = "Unknown"
#     else:
#         most_common_label, count = counter.most_common(1)[0]
#         confidence = count / len(prediction_history)
#         #print(f"Top prediction: {most_common_label}, Confidence: {confidence:.2f}")
#
#         # Threshold for confidence (e.g., must appear ≥ 40% of the time)
#         if confidence >= 0.4:
#             final_label = most_common_label
#         else:
#             final_label = "Unknown"
#
#     try:
#         name, gender, gait = final_label.split(" (")
#         gender = gender.rstrip(")")
#         gait = gait.rstrip(")")
#     except:
#         name = final_label
#         gender = "Unknown"
#         gait = "Unknown"
#
#     return {
#         "name": name,
#         "gender": gender,
#         "gait": gait,
#         # "prediction_distribution": dict(counter),  # Optional: return all counts
#         # "frames_processed": frame_count,
#         # "valid_poses": valid_pose_count,
#         # "confidence": round(confidence, 2) if counter else 0.0
#     }

# ---- this one is a little bit fast ----
# def analyze_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     pose_buffer = deque(maxlen=SEQ_LENGTH)
#     prediction_history = []
#
#     frame_count = 0
#     valid_pose_count = 0
#     frame_skip = 2  # Process every 2nd frame to speed things up
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue  # Skip this frame to reduce processing
#
#         frame = cv2.resize(frame, FRAME_SIZE)
#         landmarks, pose_landmarks = extract_pose_landmarks(frame)
#
#         if landmarks:
#             valid_pose_count += 1
#             pose_buffer.append(landmarks)
#
#             # Only run prediction if we have exactly SEQ_LENGTH
#             if len(pose_buffer) == SEQ_LENGTH:
#                 sequence = np.array(pose_buffer).flatten().reshape(1, -1)
#
#                 try:
#                     sequence_scaled = scaler.transform(sequence)
#                     pred = model.predict(sequence_scaled)
#                     pred_label = label_encoder.inverse_transform(pred)[0]
#                     prediction_history.append(pred_label)
#                 except Exception as e:
#                     continue
#
#     cap.release()
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose_buffer = deque(maxlen=SEQ_LENGTH)
    prediction_history = []

    frame_count = 0
    valid_pose_count = 0
    frame_skip = 2  # Process every 2nd frame to speed things up

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, FRAME_SIZE)
            landmarks, pose_landmarks = extract_pose_landmarks(frame, pose)

            if landmarks:
                valid_pose_count += 1
                pose_buffer.append(landmarks)

                if len(pose_buffer) == SEQ_LENGTH:
                    sequence = np.array(pose_buffer).flatten().reshape(1, -1)

                    try:
                        sequence_scaled = scaler.transform(sequence)
                        pred = model.predict(sequence_scaled)
                        pred_label = label_encoder.inverse_transform(pred)[0]
                        prediction_history.append(pred_label)
                    except Exception:
                        continue

    cap.release()


    counter = Counter(prediction_history)
    if not counter:
        final_label = "Unknown"
    else:
        most_common_label, count = counter.most_common(1)[0]
        confidence = count / len(prediction_history)
        final_label = most_common_label if confidence >= 0.4 else "Unknown"

    try:
        name, gender, gait = final_label.split(" (")
        gender = gender.rstrip(")")
        gait = gait.rstrip(")")
    except:
        name = final_label
        gender = "Unknown"
        gait = "Unknown"

    return {
        "name": name,
        "gender": gender,
        "gait": gait
    }


if __name__ == "__main__":
    main()
