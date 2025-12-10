import torch
import time

INPUT_DIM = 256
HIDDEN_DIM = 512
SEQ_LEN = 64
BATCH_SIZE = 1024
WARMUP = 10
ITERS = 100

def benchmark_rnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == 'cpu':
        print("Erreur: PyTorch n'a pas détecté de GPU. Le benchmark doit être effectué sur CUDA.")
        return

    model = torch.nn.RNN(
        input_size=INPUT_DIM,
        hidden_size=HIDDEN_DIM,
        num_layers=1,
        nonlinearity='tanh',
        batch_first=False,
    ).to(device)

    x_cpu = torch.ones(SEQ_LEN, BATCH_SIZE, INPUT_DIM, device='cpu', dtype=torch.float32)
    
    x_gpu = x_cpu.to(device)
    
    
    # ----------------------------------------------------------------------
    # TEST 1 : Performance Totale (Calcul + Copies H->D et D->H)
    # Simule l'exécution complète incluant le transfert de l'entrée et la récupération de la sortie.
    # ----------------------------------------------------------------------
    
    for _ in range(WARMUP):
        
        x_in = x_cpu.to(device, non_blocking=True)
        
        h_out, h_T = model(x_in)
        
        h_out.to('cpu', non_blocking=True)
    torch.cuda.synchronize()

    start_total = time.time()
    for _ in range(ITERS):
        x_in = x_cpu.to(device, non_blocking=True)
        h_out, h_T = model(x_in)
        h_out.to('cpu', non_blocking=True)
    torch.cuda.synchronize()
    end_total = time.time()

    avg_ms_total = (end_total - start_total) * 1000 / ITERS
    print("\n--- TEST 1 : PyTorch Performance Totale (H<->D + Calcul) ---")
    print(f"Temps moyen d'inférence RNN PyTorch TOTAL : {avg_ms_total:.4f} ms")


    # ----------------------------------------------------------------------
    # TEST 2 : Performance du Calcul Pur (GPU-Only)
    # Simule votre forward_gpu_only : l'entrée est déjà sur le GPU.
    # ----------------------------------------------------------------------


    for _ in range(WARMUP):
        h_out, h_T = model(x_gpu)
    torch.cuda.synchronize()

    start_gpu = time.time()
    for _ in range(ITERS):
        
        h_out, h_T = model(x_gpu)
    torch.cuda.synchronize()
    end_gpu = time.time()

    avg_ms_gpu_only = (end_gpu - start_gpu) * 1000 / ITERS
    print("\n--- TEST 2 : PyTorch Performance Calcul Pur (GPU-Only) ---")
    print(f"Temps moyen d'inférence RNN PyTorch GPU-ONLY : {avg_ms_gpu_only:.4f} ms")
    
    # ----------------------------------------------------------------------
    # Analyse
    # ----------------------------------------------------------------------
    copy_overhead_pt = avg_ms_total - avg_ms_gpu_only
    print("\n--- ANALYSE DE LA PERFORMANCE PyTorch ---")
    print(f"Latence des copies H<->D estimée par PyTorch : {copy_overhead_pt:.4f} ms")


if __name__ == "__main__":
    benchmark_rnn()