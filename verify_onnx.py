import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# Configuration
MODEL_PATH = os.path.join("web", "model.onnx")
IMAGE_PATH = os.path.join("data", "Testing", "glioma", "Te-glTr_0000.jpg")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def verify():
    print(f"Loading model from {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH)
    
    print(f"Loading image from {IMAGE_PATH}...")
    img = Image.open(IMAGE_PATH).convert('RGB')
    img = img.resize((224, 224))
    
    # Preprocessing (Match training/JS)
    img_data = np.array(img).astype(np.float32)
    img_data = img_data.transpose(2, 0, 1) # HWC -> CHW
    img_data /= 255.0 # 0-255 -> 0-1
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_data = (img_data - mean) / std
    
    # Add batch dimension
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    
    # Run Inference
    input_name = session.get_inputs()[0].name
    print(f"Input Name: {input_name}")
    
    result = session.run(None, {input_name: img_data})
    logits = result[0][0]
    probs = softmax(logits)
    
    print("\nResults:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {probs[i]:.4f}")
        
    predicted_idx = np.argmax(probs)
    print(f"\nPredicted: {CLASSES[predicted_idx]} ({probs[predicted_idx]:.4f})")

if __name__ == "__main__":
    verify()
