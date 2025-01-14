import torch

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is NOT available.")

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
