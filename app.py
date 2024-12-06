from flask import Flask, request, render_template, jsonify, redirect, url_for
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import os
import time

app = Flask(__name__)

# Configure Generative AI
genai.configure(api_key="AIzaSyDmSfEyvcVnr929QUnhkB4LUSmnf_NZSlo")
gen_model = genai.GenerativeModel("gemini-1.5-flash")  # Generative AI model

# Toxic comment classification model setup
MODEL_NAME = "unitary/toxic-bert"

if not os.path.exists('models'):
    os.makedirs('models')

try:
    tokenizer = AutoTokenizer.from_pretrained('models')
    model = AutoModelForSequenceClassification.from_pretrained('models')
except:
    print("Downloading model from Hugging Face... This might take a few minutes.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained('models')
    model.save_pretrained('models')

model.eval()  # Set model to evaluation mode

# Track user offenses by IP
user_status = {}

def predict_offensive(text):
    """Predict whether a message is offensive."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    confidence = probabilities[0][prediction].item()
    return prediction == 1, confidence  # Returns (is_offensive, confidence)

def manage_penalty(user_id):
    """Manage user penalties based on offense count."""
    if user_id not in user_status:
        user_status[user_id] = {"offense_count": 0, "penalty_time": None, "blocked": False}

    status = user_status[user_id]

    if status["blocked"]:
        return "blocked"

    if status["penalty_time"] and time.time() < status["penalty_time"]:
        return "frozen"

    status["offense_count"] += 1
    if status["offense_count"] == 1:
        return "warning"
    elif status["offense_count"] == 2:
        status["penalty_time"] = time.time() + 300  # Freeze for 5 minutes
        return "frozen"
    else:
        status["blocked"] = True
        return "blocked"

def generate_ai_response(prompt):
    """Generate a dynamic response using Google Generative AI."""
    try:
        response = gen_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "I'm sorry, I couldn't generate a response at the moment."

def generate_bot_response(status, confidence=None):
    """Generate a bot response based on user status."""
    responses = {
        "clean": lambda: generate_ai_response("Provide a helpful, polite response to a clean message."),
        "warning": lambda: f"Please be mindful of your language to keep our conversation respectful. (Confidence: {confidence:.2%})",
        "frozen": lambda: "Your account is temporarily restricted for 5 minutes due to repeated offenses.",
        "blocked": lambda: "You have been blocked from further communication due to repeated violations.",
    }
    return responses.get(status, lambda: "Thank you for your message!")()

@app.route('/')
def index():
    user_id = request.remote_addr
    if user_id in user_status and user_status[user_id]["blocked"]:
        return redirect(url_for('error'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.remote_addr
    message = request.json['message']

    is_offensive, confidence = predict_offensive(message)
    if is_offensive:
        penalty_status = manage_penalty(user_id)
        bot_response = generate_bot_response(penalty_status, confidence)
        return jsonify({
            "status": penalty_status,
            "bot_response": bot_response,
            "confidence": confidence
        })
    else:
        bot_response = generate_bot_response("clean")
        return jsonify({
            "status": "clean",
            "bot_response": bot_response
        })

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/admin')
def admin():
    blocked_users = [
        {"ip": user_id, "offense_count": status["offense_count"]}
        for user_id, status in user_status.items()
        if status["blocked"]
    ]
    return render_template('admin.html', blocked_users=blocked_users)

@app.route('/unblock/<user_ip>', methods=['POST'])
def unblock_user(user_ip):
    if user_ip in user_status and user_status[user_ip]["blocked"]:
        user_status[user_ip]["blocked"] = False
        user_status[user_ip]["offense_count"] = 0
        user_status[user_ip]["penalty_time"] = None
        return jsonify({"success": True})
    return jsonify({"success": False})

if __name__ == "__main__":
    app.run(debug=True)
