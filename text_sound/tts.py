import pyttsx3

def speak(text):
    """Convert text to speech"""
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)   # speed
    engine.setProperty("volume", 1.0) # volume
    engine.say(text)
    engine.runAndWait()
