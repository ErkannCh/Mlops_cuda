import torch
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    batch_size = 1024

    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    ).to(device)

    # Données d'entrée
    x = torch.ones(batch_size, input_dim, device=device)

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
    print(f"Temps moyen d'inférence PyTorch (batch_size={batch_size}): {avg_ms:.4f} ms")

if __name__ == "__main__":
    main()
