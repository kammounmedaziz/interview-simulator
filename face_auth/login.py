import cv2
import os
import pickle
import numpy as np
import sys
# Add parent directory to path to import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepface import DeepFace
from text_sound.tts import speak   # import TTS function

DB_FILE = "face_db.pkl"

# ------------------ Database Utils ------------------
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

# ------------------ Face Matching ------------------
def verify_face(embedding, embeddings):
    sims = [np.dot(embedding, e) / (np.linalg.norm(embedding) * np.linalg.norm(e)) for e in embeddings]
    return np.mean(sims)

# ------------------ Login ------------------
def login():
    db = load_db()
    if not db:
        speak("No users found in database. Please register first.")
        print("[ERROR] Database is empty. Register first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Error. Cannot access the camera.")
        print("[ERROR] Cannot access the camera")
        return

    speak("Please look at the camera to login.")

    authenticated_user = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Login - Press Q to quit", frame)

        try:
            embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
            embedding = embedding[0]["embedding"]

            for name, embeddings in db.items():
                score = verify_face(embedding, embeddings)
                if score > 0.7:  # similarity threshold
                    authenticated_user = name
                    break
        except:
            pass

        if authenticated_user:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if authenticated_user:
        speak(f"Welcome back, {authenticated_user}")
        print(f"[SUCCESS] Logged in as {authenticated_user}")
    else:
        speak("Login failed. User not recognized.")
        print("[ERROR] Login failed")

if __name__ == "__main__":
    login()
