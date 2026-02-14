import onnx
import os

model_path = os.path.join("web", "model.onnx")
model = onnx.load(model_path)

print(f"Model size on disk: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Check for external data
external_data = []
for tensor in model.graph.initializer:
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        external_data.append(tensor.name)

if external_data:
    print(f"FAIL: Found {len(external_data)} tensors with external data reference!")
    print(f"First 5: {external_data[:5]}")
else:
    print("SUCCESS: No external data references found. Model is self-contained.")
