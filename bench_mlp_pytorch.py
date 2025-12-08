import torch
import time

#new

def benchmark_mlp(input_dim, hidden_dim, output_dim, batch_size, iters=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    ).to(device)

    x = torch.ones(batch_size, input_dim, device=device)

    for _ in range(10):
        y = model(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        y = model(x)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000.0 / iters
    return avg_ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architectures = [
		(128,   64, 10,   32),
		(128,   64, 10,  128),
		(128,   64, 10,  512),
		(128,   64, 10, 1024),
		(128,  128, 10,   32),
		(128,  128, 10,  128),
		(128,  128, 10,  512),
		(128,  128, 10, 1024),
		(128,  256, 10,   32),
		(128,  256, 10,  128),
		(128,  256, 10,  512),
		(128,  256, 10, 1024),
		(128,  512, 10,   32),
		(128,  512, 10,  128),
		(128,  512, 10,  512),
		(128,  512, 10, 1024),
		(128, 1024, 10,   32),
		(128, 1024, 10,  128),
		(128, 1024, 10,  512),
		(128, 1024, 10, 1024),
	]


    for input_dim, hidden_dim, output_dim, batch_size in architectures:
        avg_ms = benchmark_mlp(input_dim, hidden_dim, output_dim, batch_size)
        print(
            f"  in={input_dim:4d}, hid={hidden_dim:4d}, out={output_dim:4d}, "
            f"batch={batch_size:4d} -> {avg_ms:.4f} ms"
        )


if __name__ == "__main__":
    main()
