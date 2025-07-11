from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
import librosa.display

import matplotlib
matplotlib.use('Agg')  # ← this MUST come before importing pyplot
import matplotlib.pyplot as plt

app = Flask(__name__)
VOICEPRINT_DIR = "voiceprints"
os.makedirs(VOICEPRINT_DIR, exist_ok=True)

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

@app.route("/register_voice", methods=["POST"])
def register_voice():
    user_id = request.form["user_id"]
    audio = request.files["audio"]
    save_path = os.path.join(VOICEPRINT_DIR, f"{user_id}.wav")
    audio.save(save_path)

    mfcc = extract_mfcc(save_path)
    np.save(os.path.join(VOICEPRINT_DIR, f"{user_id}.npy"), mfcc)

    return jsonify({"message": f"Voice registered for {user_id}."})

@app.route("/compare_voice", methods=["POST"])
def compare_voice():
    user_id = request.form["user_id"]
    audio = request.files["audio"]
    temp_path = os.path.join(VOICEPRINT_DIR, f"temp_{user_id}.wav")
    audio.save(temp_path)

    mfcc_attempt = extract_mfcc(temp_path)
    stored_path = os.path.join(VOICEPRINT_DIR, f"{user_id}.npy")

    if not os.path.exists(stored_path):
        return jsonify({"error": "User not found"}), 404

    mfcc_stored = np.load(stored_path)
    dist = np.linalg.norm(mfcc_attempt - mfcc_stored)
    match_score = max(0, 100 - dist / 10)
    matched = match_score > 75

    # Speak result
    if matched:
        print("✅ Matched: speaking access granted")
        os.system('say -v Alex "Voice match confirmed. Access granted."')
    else:
        print("❌ No match: speaking access denied")
        os.system('say -v Alex "Voice match failed. Access denied."')

    # Plot and save match score bar chart
    plt.figure()
    plt.bar(["Match Score"], [match_score])
    plt.ylim(0, 100)
    plt.title("Voice Authentication Match Score")
    plt.ylabel("Score")
    plt.savefig("match_score_chart.png")
    plt.close()

    # Plot waveform comparison
    try:
        y1, sr1 = librosa.load(temp_path)
        y2, sr2 = librosa.load(os.path.join(VOICEPRINT_DIR, f"{user_id}.wav"))

        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y1, sr=sr1)
        plt.title("Test Voice Waveform")

        plt.subplot(2, 1, 2)
        librosa.display.waveshow(y2, sr=sr2)
        plt.title("Registered Voice Waveform")

        plt.tight_layout()
        plt.savefig("voice_comparison.png")
        plt.close()
    except Exception as e:
        print(f"Waveform plot failed: {e}")

    return jsonify({
        "user_id": str(user_id),
        "match_score": float(round(match_score, 2)),
        "matched": bool(matched),
        "message": "Each voiceprint is based on unique vocal frequencies and MFCCs – just like a fingerprint, no two are the same."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)