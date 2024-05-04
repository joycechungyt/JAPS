from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from google.cloud import speech
import io

app = FastAPI()

def speech_recognition(content):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(f"Transcript: {result.alternatives[0].transcript}")

    return response

@app.post("/audio")
async def receive_audio(audio: UploadFile = File(...)):
    # Read the audio bytes
    audio_bytes = await audio.read()

    # Process the audio bytes (optional)
    # You can use a library like librosa or soundfile to process the audio data
    response = speech_recognition(audio_bytes)

    print(response.results)

    # Return a response
    return {"message": "Audio received successfully"}