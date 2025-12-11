import torch
import time
import os
import numpy as np
import math
from typing import List, Dict, Any, Tuple
import sys # Ajout pour le logging
import io # Ajout pour le logging

# Structure pour stocker les configurations d'architecture CNN
class CNNArchConfig:
    def __init__(self, N, C_in, H, W, C_out_conv, K, fc_out):
        self.N = N
        self.C_in = C_in
        self.H = H
        self.W = W
        self.C_out_conv = C_out_conv
        self.K = K
        # PyTorch calcule la sortie de Conv2d avec un padding implicite de 0 et un stride de 1
        self.H_out = H - K + 1
        self.W_out = W - K + 1
        self.fc_out = fc_out
        
        # Vérification rapide de la validité de la configuration
        if self.H_out <= 0 or self.W_out <= 0:
            raise ValueError(f"Configuration non valide: H={H}, W={W}, K={K}. La taille de sortie est négative ou nulle.")

    def label(self):
        return f"N={self.N}, Cin={self.C_in}, HxW={self.H}x{self.W}, Cout={self.C_out_conv}, K={self.K}, FC={self.fc_out}"

# Nom du fichier de sortie
OUTPUT_FILENAME_CNN = "pytorch_cnn_benchmark_results.txt"


# Fonction pour créer et initialiser le modèle PyTorch
def create_pytorch_model(cfg: CNNArchConfig, device: torch.device):
    # La formule pour la dimension de l'entrée du FC est :
    # C_out_conv * H_out * W_out
    fc_in_size = cfg.C_out_conv * cfg.H_out * cfg.W_out
    
    # Création du modèle CNN (Conv2d -> ReLU -> Flatten -> Linear)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(cfg.C_in, cfg.C_out_conv, kernel_size=cfg.K),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(fc_in_size, cfg.fc_out)
    ).to(device)
    
    # Mettre le modèle en mode évaluation (pas de calcul de gradients)
    model.eval()
    return model

def benchmark_cnn_pytorch(cfg: CNNArchConfig, iters: int = 100, warmup: int = 10) -> Tuple[float, float, List[float]]:
    
    # Utilisation du GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = create_pytorch_model(cfg, device)
    except ValueError as e:
        print(f"Erreur de configuration pour {cfg.label()}: {e}")
        return -1.0, -1.0, []

    # --------------------------------------------------
    # Préparation des Tensors d'entrée/sortie
    # --------------------------------------------------
    
    # Création de l'input hôte (pour simuler la source de données)
    input_host = torch.ones(cfg.N, cfg.C_in, cfg.H, cfg.W, dtype=torch.float32)

    # Création de l'input device (pour le GPU-ONLY)
    x_device = input_host.to(device)

    # --------------------------------------------------
    # Warmup
    # --------------------------------------------------
    with torch.no_grad():
        for _ in range(warmup):
            model(x_device)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # ----------------------------------------------------------------------
    # TEST 1 : Performance Totale (Simulant H->D, Calcul, D->H)
    # ----------------------------------------------------------------------
    
    start_total = time.time()
    with torch.no_grad():
        for i in range(iters):
            # Simulation H->D : Copie de l'hôte au device
            x_cuda = input_host.to(device)
            
            # Calcul sur le device
            y_cuda = model(x_cuda)
            
            # Simulation D->H : Copie du device à l'hôte
            y_host = y_cuda.to('cpu')
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_total = time.time()
    avg_total = (end_total - start_total) * 1000 / iters

    # ----------------------------------------------------------------------
    # TEST 2 : Performance du Calcul Pur (GPU-Only)
    # ----------------------------------------------------------------------
    
    start_gpu = time.time()
    with torch.no_grad():
        for i in range(iters):
            # Calcul sur le device, l'input est déjà sur le GPU
            y_cuda = model(x_device)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_gpu = time.time()
    avg_gpu = (end_gpu - start_gpu) * 1000 / iters

    # Récupération de l'output pour la validation
    # On prend la première ligne du batch
    output_validation = y_cuda[0, :cfg.fc_out].detach().cpu().numpy().tolist()
    
    return avg_total, avg_gpu, output_validation

# Nouvelle classe pour gérer l'impression sur la console et dans un fichier
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def main():
    # Enregistre le stdout vers la console ET le fichier
    original_stdout = sys.stdout
    sys.stdout = Logger(OUTPUT_FILENAME_CNN)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("!!! ATTENTION: CUDA non disponible. Le benchmark s'exécute sur le CPU. Les temps mesurés ne reflètent PAS la performance GPU. !!!\n")
    
    # --- Définition des plages de paramètres demandées ---
    N_vals = [32, 64, 128, 256, 512, 1024]
    C_in_vals = [3]
    H_W_vals = [32, 64] 
    C_out_conv_vals = [16, 32]
    K_vals = [3, 5, 7, 9]
    FC_OUT_vals = [10, 100]

    # Génération exhaustive des configurations (192 combinaisons)
    archs: List[CNNArchConfig] = []
    
    for N in N_vals:
        for C_in in C_in_vals:
            for HW in H_W_vals:
                for C_out_conv in C_out_conv_vals:
                    for K in K_vals:
                        for fc_out in FC_OUT_vals:
                            try:
                                archs.append(CNNArchConfig(N, C_in, HW, HW, C_out_conv, K, fc_out))
                            except ValueError as e:
                                print(f"Configuration ignorée: N={N}, H={HW}, K={K}. Erreur: {e}")
    
    print(f"Démarrage du benchmarking SimpleCNN sur PyTorch/CUDA avec {len(archs)} configurations (Exhaustif)...\n")
    
    # Mode unique pour PyTorch
    mode_name = "PYTORCH" 

    # Entête du tableau (format similaire au C++)
    separator = "----------------------------------------------------------------------------------------------------------------------------"
    header = "| Configuration             | Mode    | Temps TOTAL (ms) | Temps GPU-ONLY (ms) | Latence Copies (ms) |"
    
    print(separator)
    print(header)
    print(separator)

    for cfg in archs:
        # Mesure pour PyTorch
        avg_total, avg_gpu, output_validation = benchmark_cnn_pytorch(cfg)

        if avg_total < 0.0:
             # Affiche l'erreur si une configuration est impossible (mais gérée dans benchmark)
            print(f"| {cfg.label():<25} | {mode_name:<7} | {'ERREUR':16} | {'ERREUR':19} | {'ERREUR':19} |")
            print(separator)
            continue

        copy_overhead = avg_total - avg_gpu
        
        # Affichage des résultats
        print(f"| {cfg.label():<25} | {mode_name:<7} | {avg_total:16.4f} | {avg_gpu:19.4f} | {copy_overhead:19.4f} |")

        # Affichage de l'aperçu de la sortie
        output_str = " ".join([f"{x:.4f}" for x in output_validation[:5]])
        # Assure un bon alignement en utilisant un formatage de chaîne similaire à C++
        print(f"| {'':<25} | {'':<7} | (Aperçu) Premiers outputs ({mode_name}): {output_str} ")
        
        print(separator)
        
    # Restaure le stdout original et ferme le fichier de log
    if isinstance(sys.stdout, Logger):
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"\nLes résultats complets ont été enregistrés dans {OUTPUT_FILENAME_CNN}")

if __name__ == "__main__":
    main()