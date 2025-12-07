
import torch
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    input_dim = 128
    hidden_dim = 256
    seq_len = 32
    batch_size = 64

    rnn = torch.nn.RNN(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=1,
        nonlinearity="tanh",
        batch_first=False,
    ).to(device)

    x = torch.ones(seq_len, batch_size, input_dim, device=device)

    for _ in range(10):
        y, h_T = rnn(x)
    torch.cuda.synchronize()

    iters = 100
    start = time.time()
    for _ in range(iters):
        y, h_T = rnn(x)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000 / iters
    print(f"Temps moyen d'inférence RNN PyTorch (batch_size={batch_size}, seq_len={seq_len}): {avg_ms:.4f} ms")

if __name__ == "__main__":
    main()
