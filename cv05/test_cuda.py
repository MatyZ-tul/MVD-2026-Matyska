import torch

print("=== TEST CUDA A PYTORCH ===")
# 1. Kontrola dostupnosti CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA dostupná: {cuda_available}")

if cuda_available:
    # 2. Základní informace o GPU
    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Nalezená GPU: {gpu_name}")
    print(f"Compute Capability: sm_{major}{minor}")
    print(f"Verze CUDA v PyTorchi: {torch.version.cuda}")

    # 3. Krátký výpočet přímo na GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    print("Testovací maticové násobení na GPU proběhlo úspěšně!")
else:
    print("⚠️ Varování: PyTorch běží pouze na CPU.")
