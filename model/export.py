import torch
import torch.nn as nn
from torchvision import models
import torch.onnx
import os
import sys

# Force UTF-8 for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
MODEL_PATH = "brain_tumor_resnet18.pth"
ONNX_PATH = os.path.join("web", "model.onnx")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def export_model():
    # 1. Initialize Model Architecture
    model = models.resnet18(pretrained=False) # We load our own weights, so pretrained=False is fine
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    # 2. Load Weights
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run train.py first.")
        # Create a dummy model for demonstration purposes if file is missing? 
        # No, better to force the user to train or provide a pre-trained path.
        # But for the sake of the user being able to test the export logic, we might proceed with random weights 
        # if they just want to see the ONNX file generated (with a warning).
        print("Warning: Generating ONNX with random weights for testing purposes.")
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        print(f"Loaded weights from {MODEL_PATH}")

    model.eval()

    # 3. Create Dummy Input
    # Standard ImageNet input size: 1x3x224x224
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. Export to ONNX
    print(f"Exporting model to {ONNX_PATH}...")
    torch.onnx.export(model,               # model being run
                      dummy_input,         # model input (or a tuple for multiple inputs)
                      ONNX_PATH,           # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    
    # Check if file was split
    if os.path.exists(ONNX_PATH + ".data"):
        print("WARNING: Model was split into .data file! This will cause issues in browser.")
    else:
        size = os.path.getsize(ONNX_PATH)
        print(f"Success! Model exported as single file. Size: {size / 1024 / 1024:.2f} MB")
    
    print("Model exported successfully!")
    
    # Attempt to repackage if split
    if os.path.exists(ONNX_PATH + ".data"):
        print("Attempting to repackage split model into single file...")
        try:
            import onnx
            # Load the model (this automatically loads external data if present)
            onnx_model = onnx.load(ONNX_PATH)
            
            # Save it back to the same path, but force no external data
            # The default behavior of onnx.save is to inline data if it fits (protobuf limit 2GB)
            # We explicitly ensure it doesn't try to use external data
            onnx.save_model(onnx_model, ONNX_PATH)
            
            # Clean up .data file
            if os.path.exists(ONNX_PATH + ".data"):
                os.remove(ONNX_PATH + ".data")
                
            print(f"Repackaged successfully! Final size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.2f} MB")
        except ImportError:
            print("Error: 'onnx' library not found. Cannot repackage. Please install via 'pip install onnx'")
        except Exception as e:
            print(f"Failed to repackage: {e}")

    print("You can now load this model in the web interface.")

if __name__ == '__main__':
    export_model()
