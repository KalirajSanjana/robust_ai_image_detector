from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import io

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

app = Flask(__name__)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"

# Change this filename only if your actual .pth filename is different.
MODEL_PATH = TEST_DIR / "resnet18_1000perclass_20epochs_combinedataset_with_50_transform.pth"

# ------------------------------------------------------------
# Model setup
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

class_names = {
    0: "REAL",
    1: "AI"
}

print(f"Using device: {device}")
print(f"Loaded model: {MODEL_PATH}")


# ------------------------------------------------------------
# Serve index.html
# ------------------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/logo.jpg")
def logo():
    return send_from_directory(BASE_DIR, "logo.jpg")

# ------------------------------------------------------------
# Detect uploaded images
# ------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def detect():
    uploaded_files = request.files.getlist("images")

    if not uploaded_files:
        return jsonify({"error": "No images uploaded"}), 400

    results = {}

    with torch.no_grad():
        for uploaded_file in uploaded_files:
            filename = uploaded_file.filename or "image"

            try:
                # Read uploaded image directly from memory
                image_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Same preprocessing as your original script
                input_tensor = transform(image)
                input_tensor = input_tensor.unsqueeze(0).to(device)

                # Run model
                outputs = model(input_tensor)

                # Convert logits to probabilities
                probabilities = torch.softmax(outputs, dim=1)[0]

                # Get highest-probability class
                confidence, class_id = torch.max(probabilities, dim=0)

                class_id = int(class_id.item())
                confidence = float(confidence.item())
                predicted_class = class_names[class_id]

                results[filename] = {
                    "predicted_class_id": class_id,
                    "predicted_class": predicted_class,
                    "confidence_score": confidence
                }

                print(
                    f"{filename} -> "
                    f"{predicted_class} "
                    f"({confidence:.2%})"
                )

            except Exception as e:
                results[filename] = {
                    "error": str(e)
                }

    return jsonify(results)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
