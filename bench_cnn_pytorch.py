import torch
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 64
    C_in = 1
    H = 28
    W = 28
    C_out_conv = 8
    K = 3
    fc_out = 10

    model = torch.nn.Sequential(
        torch.nn.Conv2d(C_in, C_out_conv, kernel_size=K, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(C_out_conv * (H - K + 1) * (W - K + 1), fc_out),
    ).to(device)

    x = torch.ones(N, C_in, H, W, device=device)

    # Warmup
    for _ in range(10):
        y = model(x)
    torch.cuda.synchronize()

    iters = 100
    start = time.time()
    for _ in range(iters):
        y = model(x)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000 / iters
    print(f"Temps moyen d'inférence CNN PyTorch (batch_size={N}): {avg_ms:.4f} ms")

if __name__ == "__main__":
    main()
