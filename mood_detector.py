
from transformers import pipeline

#  Emotion model: detects happy/sad
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True)

def predict_mood(text):
    # Emotion model prediction
    emo_results = emotion_model(text)[0]
    emo_scores = {item["label"]: item["score"] for item in emo_results}



    #  Suicidal detection (highest priority)
    suicidal_signals = [
        "hopelessness",
        "depression",
        "severe_depression",
        "self-harm"
    ]


    #  Happy detection
    if emo_scores.get("joy", 0) > 0.40:
        return "happy"

    # Sad detection
    if emo_scores.get("sadness", 0) > 0.40:
        return "sad"

    # Default
    return "neutral"

# Test script
if __name__ == "__main__":
    while True:
        text = input("\nEnter a journal entry (or 'quit'): ")
        if text.lower() == "quit":
            break
        mood = predict_mood(text)
        print(f"Predicted Mood: {mood}")

