import torch
import time
import csv
from typing import List, Dict, Tuple
from collections import namedtuple
import os # Ajout pour les opérations de fichier

# Définition de la structure pour les configurations d'architecture
RNNArchConfig = namedtuple('RNNArchConfig', ['input_dim', 'hidden_dim', 'seq_len', 'batch_size'])

WARMUP = 10
ITERS = 100
OUTPUT_FILENAME = "pytorch_rnn_benchmark_results.txt"

def benchmark_rnn_pytorch(config: RNNArchConfig, device: torch.device) -> Tuple[float, float]:
    """
    Mesure le temps d'inférence moyen du RNN PyTorch (Total et GPU-Only) pour une configuration donnée.
    
    Retourne : (temps_total_ms, temps_gpu_only_ms)
    """
    input_dim = config.input_dim
    hidden_dim = config.hidden_dim
    seq_len = config.seq_len
    batch_size = config.batch_size
    
    # Initialisation du modèle
    model = torch.nn.RNN(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=1,
        nonlinearity='tanh',
        batch_first=False,
    ).to(device)
    
    # Création des tenseurs d'entrée
    x_cpu = torch.ones(seq_len, batch_size, input_dim, device='cpu', dtype=torch.float32)
    x_gpu = x_cpu.to(device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # --- TEST 1 : Performance Totale (Calcul + Copies H->D et D->H) ---
    
    # Warmup
    for _ in range(WARMUP):
        x_in = x_cpu.to(device, non_blocking=True)
        h_out, h_T = model(x_in)
        h_out.to('cpu', non_blocking=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_total = time.time()
    for _ in range(ITERS):
        x_in = x_cpu.to(device, non_blocking=True)
        h_out, h_T = model(x_in)
        h_out.to('cpu', non_blocking=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_total = time.time()

    avg_ms_total = (end_total - start_total) * 1000 / ITERS
    
    # --- TEST 2 : Performance du Calcul Pur (GPU-Only) ---

    # Warmup
    for _ in range(WARMUP):
        h_out, h_T = model(x_gpu)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_gpu = time.time()
    for _ in range(ITERS):
        h_out, h_T = model(x_gpu)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_gpu = time.time()

    avg_ms_gpu_only = (end_gpu - start_gpu) * 1000 / ITERS
    
    return avg_ms_total, avg_ms_gpu_only

def log_result(config: RNNArchConfig, total_ms: float, gpu_only_ms: float, copy_ms: float, filename: str):
    """ Écrit le résultat du benchmark dans un fichier CSV. """
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writerow(["D_Input", "H_Hidden", "T_SeqLen", "N_Batch", "Time_Total_ms", "Time_GPU_ONLY_ms", "Latency_Copies_ms"])
        
        writer.writerow([
            config.input_dim, config.hidden_dim, config.seq_len, config.batch_size,
            f"{total_ms:.4f}", f"{gpu_only_ms:.4f}", f"{copy_ms:.4f}"
        ])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == 'cpu':
        print("\nERREUR: PyTorch n'a pas détecté de GPU. Le benchmark doit être effectué sur CUDA.")
        return

    archs: List[RNNArchConfig] = [
        # Les paramètres sont dans l'ordre: (D_in, H_hidden, T, N)
        
        # ==================================================================
        # 1. T = 32 (Séquence la plus courte)
        # ==================================================================

        # H = 128 (Petit Hidden Size)
        RNNArchConfig(256, 128, 32, 32), RNNArchConfig(256, 128, 32, 64), RNNArchConfig(256, 128, 32, 128), RNNArchConfig(256, 128, 32, 256), RNNArchConfig(256, 128, 32, 512), RNNArchConfig(256, 128, 32, 1024),
        # H = 256
        RNNArchConfig(256, 256, 32, 32), RNNArchConfig(256, 256, 32, 64), RNNArchConfig(256, 256, 32, 128), RNNArchConfig(256, 256, 32, 256), RNNArchConfig(256, 256, 32, 512), RNNArchConfig(256, 256, 32, 1024),
        # H = 512
        RNNArchConfig(256, 512, 32, 32), RNNArchConfig(256, 512, 32, 64), RNNArchConfig(256, 512, 32, 128), RNNArchConfig(256, 512, 32, 256), RNNArchConfig(256, 512, 32, 512), RNNArchConfig(256, 512, 32, 1024),
        # H = 1024 (Grand Hidden Size)
        RNNArchConfig(256, 1024, 32, 32), RNNArchConfig(256, 1024, 32, 64), RNNArchConfig(256, 1024, 32, 128), RNNArchConfig(256, 1024, 32, 256), RNNArchConfig(256, 1024, 32, 512), RNNArchConfig(256, 1024, 32, 1024),

        # ==================================================================
        # 2. T = 64
        # ==================================================================

        # H = 128
        RNNArchConfig(256, 128, 64, 32), RNNArchConfig(256, 128, 64, 64), RNNArchConfig(256, 128, 64, 128), RNNArchConfig(256, 128, 64, 256), RNNArchConfig(256, 128, 64, 512), RNNArchConfig(256, 128, 64, 1024),
        # H = 256
        RNNArchConfig(256, 256, 64, 32), RNNArchConfig(256, 256, 64, 64), RNNArchConfig(256, 256, 64, 128), RNNArchConfig(256, 256, 64, 256), RNNArchConfig(256, 256, 64, 512), RNNArchConfig(256, 256, 64, 1024),
        # H = 512
        RNNArchConfig(256, 512, 64, 32), RNNArchConfig(256, 512, 64, 64), RNNArchConfig(256, 512, 64, 128), RNNArchConfig(256, 512, 64, 256), RNNArchConfig(256, 512, 64, 512), RNNArchConfig(256, 512, 64, 1024),
        # H = 1024
        RNNArchConfig(256, 1024, 64, 32), RNNArchConfig(256, 1024, 64, 64), RNNArchConfig(256, 1024, 64, 128), RNNArchConfig(256, 1024, 64, 256), RNNArchConfig(256, 1024, 64, 512), RNNArchConfig(256, 1024, 64, 1024),

        # H = 128
        RNNArchConfig(256, 128, 128, 32), RNNArchConfig(256, 128, 128, 64), RNNArchConfig(256, 128, 128, 128), RNNArchConfig(256, 128, 128, 256), RNNArchConfig(256, 128, 128, 512), RNNArchConfig(256, 128, 128, 1024),
        # H = 256
        RNNArchConfig(256, 256, 128, 32), RNNArchConfig(256, 256, 128, 64), RNNArchConfig(256, 256, 128, 128), RNNArchConfig(256, 256, 128, 256), RNNArchConfig(256, 256, 128, 512), RNNArchConfig(256, 256, 128, 1024),
        # H = 512
        RNNArchConfig(256, 512, 128, 32), RNNArchConfig(256, 512, 128, 64), RNNArchConfig(256, 512, 128, 128), RNNArchConfig(256, 512, 128, 256), RNNArchConfig(256, 512, 128, 512), RNNArchConfig(256, 512, 128, 1024),
        # H = 1024
        RNNArchConfig(256, 1024, 128, 32), RNNArchConfig(256, 1024, 128, 64), RNNArchConfig(256, 1024, 128, 128), RNNArchConfig(256, 1024, 128, 256), RNNArchConfig(256, 1024, 128, 512), RNNArchConfig(256, 1024, 128, 1024),

        # ==================================================================
        # 4. T = 256
        # ==================================================================

        # H = 128
        RNNArchConfig(256, 128, 256, 32), RNNArchConfig(256, 128, 256, 64), RNNArchConfig(256, 128, 256, 128), RNNArchConfig(256, 128, 256, 256), RNNArchConfig(256, 128, 256, 512), RNNArchConfig(256, 128, 256, 1024),
        # H = 256
        RNNArchConfig(256, 256, 256, 32), RNNArchConfig(256, 256, 256, 64), RNNArchConfig(256, 256, 256, 128), RNNArchConfig(256, 256, 256, 256), RNNArchConfig(256, 256, 256, 512), RNNArchConfig(256, 256, 256, 1024),
        # H = 512
        RNNArchConfig(256, 512, 256, 32), RNNArchConfig(256, 512, 256, 64), RNNArchConfig(256, 512, 256, 128), RNNArchConfig(256, 512, 256, 256), RNNArchConfig(256, 512, 256, 512), RNNArchConfig(256, 512, 256, 1024),
        # H = 1024
        RNNArchConfig(256, 1024, 256, 32), RNNArchConfig(256, 1024, 256, 64), RNNArchConfig(256, 1024, 256, 128), RNNArchConfig(256, 1024, 256, 256), RNNArchConfig(256, 1024, 256, 512), RNNArchConfig(256, 1024, 256, 1024),

        # ==================================================================
        # 5. T = 512 (Séquence la plus longue)
        # ==================================================================

        # H = 128
        RNNArchConfig(256, 128, 512, 32), RNNArchConfig(256, 128, 512, 64), RNNArchConfig(256, 128, 512, 128), RNNArchConfig(256, 128, 512, 256), RNNArchConfig(256, 128, 512, 512), RNNArchConfig(256, 128, 512, 1024),
        # H = 256
        RNNArchConfig(256, 256, 512, 32), RNNArchConfig(256, 256, 512, 64), RNNArchConfig(256, 256, 512, 128), RNNArchConfig(256, 256, 512, 256), RNNArchConfig(256, 256, 512, 512), RNNArchConfig(256, 256, 512, 1024),
        # H = 512
        RNNArchConfig(256, 512, 512, 32), RNNArchConfig(256, 512, 512, 64), RNNArchConfig(256, 512, 512, 128), RNNArchConfig(256, 512, 512, 256), RNNArchConfig(256, 512, 512, 512), RNNArchConfig(256, 512, 512, 1024),
        # H = 1024
        RNNArchConfig(256, 1024, 512, 32), RNNArchConfig(256, 1024, 512, 64), RNNArchConfig(256, 1024, 512, 128), RNNArchConfig(256, 1024, 512, 256), RNNArchConfig(256, 1024, 512, 512), RNNArchConfig(256, 1024, 512, 1024),
    ]
    
    print(f"\nDémarrage du benchmarking PyTorch RNN sur CUDA avec {len(archs)} configurations...")
    print(f"Warmup: {WARMUP} itérations, Mesure: {ITERS} itérations.")
    print(f"Les résultats détaillés seront enregistrés dans {OUTPUT_FILENAME}\n")

    # Entête du tableau (pour la console)
    header = "| {:^9} | {:^10} | {:^10} | {:^9} | {:^18} | {:^19} | {:^19} |".format(
        "D (Input)", "H (Hidden)", "T (SeqLen)", "N (Batch)", 
        "Temps TOTAL (ms)", "Temps GPU-ONLY (ms)", "Latence Copies (ms)"
    )
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)

    results = []
    # Supprimer l'ancien fichier de log
    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)
        
    for config in archs:
        avg_ms_total, avg_ms_gpu_only = benchmark_rnn_pytorch(config, device)
        
        if avg_ms_total < 0:
            copy_overhead_pt = -1.0
            row = "| {:^9} | {:^10} | {:^10} | {:^9} | {:^18} | {:^19} | {:^19} |".format(
                config.input_dim, config.hidden_dim, config.seq_len, config.batch_size, 
                "ERREUR", "ERREUR", "ERREUR"
            )
        else:
            copy_overhead_pt = avg_ms_total - avg_ms_gpu_only
            row = "| {:^9} | {:^10} | {:^10} | {:^9} | {:^18.4f} | {:^19.4f} | {:^19.4f} |".format(
                config.input_dim, config.hidden_dim, config.seq_len, config.batch_size, 
                avg_ms_total, avg_ms_gpu_only, copy_overhead_pt
            )
        
        # Écriture dans la console et dans le fichier
        print(row)
        log_result(config, avg_ms_total, avg_ms_gpu_only, copy_overhead_pt, OUTPUT_FILENAME)
        results.append((config, avg_ms_total, avg_ms_gpu_only))

    print(separator)
    
    # Affichage de l'output de validation (dernière configuration)
    if results:
        last_config = archs[-1]
        
        # Re-initialisation du modèle et des tenseurs pour l'affichage de l'output
        model = torch.nn.RNN(
            input_size=last_config.input_dim,
            hidden_size=last_config.hidden_dim,
            num_layers=1,
            nonlinearity='tanh',
            batch_first=False,
        ).to(device)
        
        x_cpu = torch.ones(last_config.seq_len, last_config.batch_size, last_config.input_dim, device='cpu', dtype=torch.float32)
        x_gpu = x_cpu.to(device)
        
        # Effectuer une dernière passe
        h_out, h_T = model(x_gpu) 
        
        preview_output = f"\n[Aperçu pour la dernière configuration (D={last_config.input_dim}, H={last_config.hidden_dim}, T={last_config.seq_len}, N={last_config.batch_size})]\n"
        
        # Le dernier état caché h_T est un tenseur (1, N, H)
        output_sample = h_T[0, 0, :5].cpu().tolist()
        
        preview_output += "Premiers outputs de l'état caché final (h_T[0, 0, :5]): "
        preview_output += " ".join([f"{v:.4f}" for v in output_sample])
        preview_output += "\n"

        # Écrire l'aperçu dans la console et le fichier
        print(preview_output)
        with open(OUTPUT_FILENAME, 'a', newline='') as f:
            f.write(preview_output)


if __name__ == "__main__":
    main()