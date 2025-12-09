import re
import subprocess
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

def run_command(cmd, timeout=300):
    """
    Run a shell command and return its stdout as text.
    On error, display a Streamlit message and return None.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        st.error(f"Commande trop longue à s'exécuter: {' '.join(cmd)}")
    except subprocess.CalledProcessError as e:
        st.error(
            f"Erreur lors de l'exécution de {' '.join(cmd)} "
            f"(code {e.returncode}).\n\nstdout:\n{e.stdout}\n\nstderr:\n{e.stderr}"
        )
    except Exception as e:
        st.error(f"Erreur inattendue lors de l'exécution de {' '.join(cmd)}: {e}")
    return None


def parse_mlp_lines(stdout: str, label: str):
    pattern = re.compile(
        r"in=\s*(\d+)\s*,\s*hid=\s*(\d+)\s*,\s*out=\s*(\d+)\s*,\s*batch=\s*(\d+)\s*->\s*([0-9.eE+-]+)"
    )
    rows = []
    for line in stdout.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        input_dim, hidden_dim, output_dim, batch_size, time_ms = m.groups()
        rows.append(
            {
                "input_dim": int(input_dim),
                "hidden_dim": int(hidden_dim),
                "output_dim": int(output_dim),
                "batch_size": int(batch_size),
                f"time_ms_{label}": float(time_ms),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def section_mlp():
    st.header("Benchmarks MLP (CUDA vs PyTorch)")

    st.markdown(
        """
Cette section compare les temps d'inférence d'un MLP implémenté :

- en CUDA (`cuda_mlp`)
- en PyTorch (`bench_mlp_pytorch.py`)
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        cuda_exec_exists = Path("./build/cuda_mlp").is_file()
        st.write(f"Exécutable CUDA présent : {'✅' if cuda_exec_exists else '❌'}")
    with col2:
        pyt_script_exists = Path("./bench_mlp_pytorch.py").is_file()
        st.write(
            f"Script PyTorch présent : {'✅' if pyt_script_exists else '❌'}"
        )

    if not cuda_exec_exists:
        st.warning(
            "L'exécutable `cuda_mlp` est introuvable. "
            "Compile le projet avec CMake (`cmake --build .`) avant de lancer les benchmarks."
        )
    if not pyt_script_exists:
        st.warning(
            "Le script `bench_mlp_pytorch.py` est introuvable à la racine du projet."
        )

    if not (cuda_exec_exists and pyt_script_exists):
        return

    # Initialiser l'état la première fois
    if "mlp_df" not in st.session_state:
        st.session_state.mlp_df = None

    # Bouton : on (re)lance les benchmarks et on stocke les résultats
    if st.button("Lancer les benchmarks MLP"):
        with st.spinner("Exécution des benchmarks CUDA MLP..."):
            cuda_out = run_command(["./build/cuda_mlp"])
        with st.spinner("Exécution des benchmarks PyTorch MLP..."):
            torch_out = run_command([sys.executable, "./bench_mlp_pytorch.py"])

        if cuda_out and torch_out:
            df_cuda = parse_mlp_lines(cuda_out, "cuda")
            df_torch = parse_mlp_lines(torch_out, "torch")

            if df_cuda.empty or df_torch.empty:
                st.warning(
                    "Impossible de parser les résultats MLP (format inattendu). "
                    "Vérifie la sortie des programmes."
                )
            else:
                df = pd.merge(
                    df_cuda,
                    df_torch,
                    on=["input_dim", "hidden_dim", "output_dim", "batch_size"],
                    how="outer",
                )
                if "time_ms_cuda" in df.columns and "time_ms_torch" in df.columns:
                    df["speedup (torch/cuda)"] = (
                        df["time_ms_torch"] / df["time_ms_cuda"]
                    )
                # On stocke le df dans la session
                st.session_state.mlp_df = df

    # Partie affichage : indépendante du bouton
    df = st.session_state.mlp_df
    if df is not None:
        st.subheader("Tableau comparatif")
        st.dataframe(df.sort_values(["hidden_dim", "batch_size"]))

        st.subheader("Visualisation pour un batch_size donné")
        batch_values = sorted(df["batch_size"].dropna().unique())
        if batch_values:
            selected_batch = st.selectbox(
                "Choisis un batch_size pour la vue ci-dessous",
                batch_values,
                index=0,
            )
            df_batch = df[df["batch_size"] == selected_batch].copy()
            df_batch["hidden_dim_str"] = df_batch["hidden_dim"].astype(str)
            chart_data = df_batch.set_index("hidden_dim_str")[
                ["time_ms_cuda", "time_ms_torch"]
            ]
            st.bar_chart(chart_data)
    else:
        st.info("Clique sur **Lancer les benchmarks MLP** pour afficher les résultats.")



def parse_single_time_line(stdout: str, keyword: str):
    """
    Extract a single '...: xxx ms' line for CNN/RNN.
    Returns the time as float or None.
    """
    pattern = re.compile(rf"{keyword}.*:\s*([0-9.]+)\s*ms")
    for line in stdout.splitlines():
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    return None


def section_cnn():
    st.header("Benchmark CNN (CUDA vs PyTorch)")

    st.markdown(
        """
Cette section lance :

- `cuda_cnn` (implémentation CUDA du CNN)
- `bench_cnn_pytorch.py` (implémentation PyTorch équivalente)
        """
    )

    cuda_exec_exists = Path("./build/cuda_cnn").is_file()
    pyt_script_exists = Path("./bench_cnn_pytorch.py").is_file()
    if not cuda_exec_exists:
        st.warning(
            "L'exécutable `cuda_cnn` est introuvable. "
            "Compile le projet avec CMake (`cmake --build .`) avant de lancer le benchmark."
        )
    if not pyt_script_exists:
        st.warning(
            "Le script `bench_cnn_pytorch.py` est introuvable à la racine du projet."
        )

    if not (cuda_exec_exists and pyt_script_exists):
        return

    if st.button("Lancer le benchmark CNN"):
        with st.spinner("Exécution du CNN CUDA..."):
            cuda_out = run_command(["./src/cnn"])
        with st.spinner("Exécution du CNN PyTorch..."):
            torch_out = run_command([sys.executable, "./bench_cnn_pytorch.py"])

        if cuda_out and torch_out:
            t_cuda = _single_time_line(
                cuda_out, "Temps moyen d'inférence CNN"
            )
            t_torch = parse_single_time_line(
                torch_out, "Temps moyen d'inférence CNN PyTorch"
            )

            if t_cuda is None or t_torch is None:
                st.warning(
                    "Impossible de parser les temps CNN (format inattendu)."
                )
                return

            st.subheader("Résumé")
            df = pd.DataFrame(
                {
                    "implémentation": ["CUDA", "PyTorch"],
                    "temps_ms": [t_cuda, t_torch],
                }
            )
            st.dataframe(df)
            st.bar_chart(df.set_index("implémentation"))


def section_rnn():
    st.header("Benchmark RNN (CUDA vs PyTorch)")

    st.markdown(
        """
Cette section lance :

- `cuda_rnn` (implémentation CUDA du RNN)
- `bench_rnn_pytroch.py` (implémentation PyTorch équivalente)
        """
    )

    cuda_exec_exists = Path("./build/cuda_rnn").is_file()
    pyt_script_exists = Path("./bench_rnn_pytorch.py").is_file()

    st.write(f"Exécutable CUDA présent : {'✅' if cuda_exec_exists else '❌'}")
    st.write(
        f"Script PyTorch présent : {'✅' if pyt_script_exists else '❌'}"
    )

    if not cuda_exec_exists:
        st.warning(
            "L'exécutable `cuda_rnn` est introuvable. "
            "Compile le projet avec CMake (`cmake --build .`) avant de lancer le benchmark."
        )
    if not pyt_script_exists:
        st.warning(
            "Le script `bench_rnn_pytroch.py` est introuvable à la racine du projet."
        )

    if not (cuda_exec_exists and pyt_script_exists):
        return

    if st.button("Lancer le benchmark RNN"):
        with st.spinner("Exécution du RNN CUDA..."):
            cuda_out = run_command(["./src/rnn"])
        with st.spinner("Exécution du RNN PyTorch..."):
            torch_out = run_command(
                [sys.executable, "./bench_rnn_pytorch.py"]
            )

        if cuda_out:
            st.subheader("Sortie brute CUDA")
            st.code(cuda_out)
        if torch_out:
            st.subheader("Sortie brute PyTorch")
            st.code(torch_out)

        if cuda_out and torch_out:
            t_cuda = parse_single_time_line(
                cuda_out, "Temps moyen d'inférence RNN"
            )
            t_torch = parse_single_time_line(
                torch_out, "Temps moyen d'inférence RNN PyTorch"
            )

            if t_cuda is None or t_torch is None:
                st.warning(
                    "Impossible de parser les temps RNN (format inattendu)."
                )
                return

            st.subheader("Résumé")
            df = pd.DataFrame(
                {
                    "implémentation": ["CUDA", "PyTorch"],
                    "temps_ms": [t_cuda, t_torch],
                }
            )
            st.dataframe(df)
            st.bar_chart(df.set_index("implémentation"))


def main():
    st.set_page_config(
        page_title="Benchmarks CUDA / PyTorch",
        layout="wide",
    )

    st.sidebar.title("Menu")
    section = st.sidebar.radio(
        "Choisis une section",
        ("MLP", "CNN", "RNN"),
    )

    if section == "MLP":
        section_mlp()
    elif section == "CNN":
        section_cnn()
    elif section == "RNN":
        section_rnn()


if __name__ == "__main__":
    main()

