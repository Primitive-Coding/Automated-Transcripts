# NLP
import speech_recognition as sr


def use_google_transcriptions(path_to_file: str) -> str:
    rec = sr.Recognizer()
    with sr.AudioFile(path_to_file) as source:
        audio_listened = rec.record(source)
        # try converting it to text
        try:
            text = rec.recognize_google(audio_listened)
        except sr.UnknownValueError as e:
            text = "Error"
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(path_to_file, ":", text)
    return text


def use_whisper_transcriptions(path_to_file) -> str:
    rec = sr.Recognizer()
    with sr.AudioFile(path_to_file) as source:
        audio_listened = rec.record(source)
        # try converting it to text
        try:
            text = rec.recognize_whisper(audio_listened)
        except sr.UnknownValueError as e:
            text = "Error"
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(path_to_file, ":", text)
    return text


def use_amazon_transcriptions(path_to_file) -> str:
    rec = sr.Recognizer()
    with sr.AudioFile(path_to_file) as source:
        audio_listened = rec.record(source)
        # try converting it to text
        try:
            text = rec.recognize_amazon(audio_listened)
        except sr.UnknownValueError as e:
            text = "Error"
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(path_to_file, ":", text)
    return text


def use_ibm_transcriptions(path_to_file) -> str:
    rec = sr.Recognizer()
    with sr.AudioFile(path_to_file) as source:
        audio_listened = rec.record(source)
        # try converting it to text
        try:
            text = rec.recognize_ibm(audio_listened)
        except sr.UnknownValueError as e:
            text = "Error"
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(path_to_file, ":", text)
    return text


def use_azure_transcriptions(path_to_file) -> str:
    rec = sr.Recognizer()
    with sr.AudioFile(path_to_file) as source:
        audio_listened = rec.record(source)
        # try converting it to text
        try:
            text = rec.recognize_azure(audio_listened)
        except sr.UnknownValueError as e:
            text = "Error"
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(path_to_file, ":", text)
    return text
