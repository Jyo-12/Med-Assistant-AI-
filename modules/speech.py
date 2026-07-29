"""
modules/speech.py

Voice Assistant for MedAssist AI
Supports:
- Speech to Text
- Text to Speech
"""

import speech_recognition as sr
import pyttsx3


class SpeechAssistant:

    def __init__(self):

        self.recognizer = sr.Recognizer()

        self.tts_engine = pyttsx3.init()

        self.tts_engine.setProperty("rate", 170)

        self.tts_engine.setProperty("volume", 1.0)

    # -----------------------------------------------------

    def listen(self, timeout=5, phrase_time_limit=20):

        with sr.Microphone() as source:

            print("Adjusting for ambient noise...")

            self.recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            print("Listening...")

            audio = self.recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )

        try:

            text = self.recognizer.recognize_google(audio)

            return {
                "success": True,
                "text": text,
                "error": None
            }

        except sr.UnknownValueError:

            return {
                "success": False,
                "text": "",
                "error": "Speech could not be understood."
            }

        except sr.RequestError as e:

            return {
                "success": False,
                "text": "",
                "error": str(e)
            }

    # -----------------------------------------------------

    def speech_to_text(self):

        result = self.listen()

        if result["success"]:

            return result["text"]

        return ""

    # -----------------------------------------------------

    def text_to_speech(self, text):

        self.tts_engine.say(text)

        self.tts_engine.runAndWait()

    # -----------------------------------------------------

    def speak(self, text):

        self.text_to_speech(text)


if __name__ == "__main__":

    assistant = SpeechAssistant()

    print("Say something...")

    spoken = assistant.speech_to_text()

    print("You said:", spoken)

    assistant.text_to_speech(
        "Hello. Your MedAssist AI Voice Assistant is working correctly."
    )