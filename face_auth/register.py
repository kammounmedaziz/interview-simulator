import cv2
import os
import pickle
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

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

# ------------------ Registration ------------------
def register_user(name):
    db = load_db()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Error. Cannot access the camera.")
        print("[ERROR] Cannot access the camera")
        return

    speak("Welcome to the app. It's time to register, " + name)

    embeddings = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Register - Press Q to quit", frame)

        if count < 10:
            try:
                embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
                embeddings.append(embedding[0]["embedding"])
                count += 1
                print(f"[INFO] Captured {count}/10 frames for {name}")
            except Exception as e:
                print(f"[WARNING] Skipping frame: {e}")

        if count >= 10:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if embeddings:
        db[name] = embeddings
        save_db(db)
        speak(f"Registration successful. Welcome, {name}")
        print(f"[SUCCESS] {name} registered successfully")
    else:
        speak(f"Registration failed for {name}")
        print("[ERROR] Registration failed")

if __name__ == "__main__":
    user_name = input("Enter your name for registration: ")
    register_user(user_name)
