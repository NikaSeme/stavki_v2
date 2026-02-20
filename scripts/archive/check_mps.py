
import torch
import platform

print(f"Platform: {platform.system()} {platform.release()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")
