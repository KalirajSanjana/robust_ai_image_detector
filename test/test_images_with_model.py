import os
import json

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


def run_inference(image_folder, output_json):

    # 1. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Initialize the exact same ResNet18 architecture
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # 3. Load your trained model
    model.load_state_dict(
        torch.load(
            r"C:\Users\r7lia\Downloads\Robust AI Image Detector\test\resnet18_1000perclass_20epochs_combinedataset_with_50_transform.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    # 4. Use the EXACT same preprocessing as training
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    # 5. Class mapping from your training code
    class_names = {
        0: "REAL",
        1: "AI"
    }

    results = {}

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    # 6. Process all images in the folder
    with torch.no_grad():

        for filename in os.listdir(image_folder):

            if not filename.lower().endswith(valid_extensions):
                continue

            img_path = os.path.join(image_folder, filename)

            try:
                # Load image
                image = Image.open(img_path).convert("RGB")

                # Apply same preprocessing used during training
                input_tensor = transform(image)

                # Add batch dimension
                input_tensor = input_tensor.unsqueeze(0)

                # Move to GPU/CPU
                input_tensor = input_tensor.to(device)

                # Run model
                outputs = model(input_tensor)

                # Convert logits to probabilities
                probabilities = torch.softmax(outputs, dim=1)[0]

                # Get prediction
                confidence, class_id = torch.max(
                    probabilities,
                    dim=0
                )

                class_id = int(class_id.item())
                confidence = float(confidence.item())

                predicted_class = class_names[class_id]

                # Store results
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
                print(f"Error processing {filename}: {e}")

    # 7. Save results
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print("\nInference complete.")
    print(f"Results saved to: {output_json}")


if __name__ == "__main__":

    run_inference(
        image_folder=r"C:\Users\r7lia\Downloads\Robust AI Image Detector\test\images",
        output_json=r"C:\Users\r7lia\Downloads\Robust AI Image Detector\test\predictions.json"
    )