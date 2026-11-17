import os
import torch

print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

try:
    x = torch.randn(5, 5, device="cuda")
    y = x @ x
    print("Success! Tensor operation on RTX 5060 Ti:")
    print(y)
except Exception as e:
    print("CUDA Error:", e)
