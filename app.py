import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows frontend to communicate with backend

# Load the trained model
model = tf.keras.models.load_model("transaction_model.keras")

# Load the tokenizer and scaler
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define category labels (must match training order)
CATEGORIES = ["Clothing", "Food", "Health", "Other", "Rent", "Shopping", "Travel"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Process text input
        text_sequence = tokenizer.texts_to_sequences([data["text"]])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=50)

        # Process numerical input
        num_features = np.array(data["numerical_features"]).reshape(1, -1)
        num_scaled = scaler.transform(num_features)

        # Reshape numerical input for CNN
        num_scaled = num_scaled.reshape(1, 3, 1)

        # Make prediction
        prediction = model.predict([text_padded, num_scaled])
        predicted_index = prediction.argmax()

        print(f"Predicted Index: {predicted_index}, Predicted Category: {CATEGORIES[predicted_index]}")

        return jsonify({"category": CATEGORIES[predicted_index]})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
