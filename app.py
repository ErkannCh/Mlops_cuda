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
    main_left_col, main_right_col = st.columns([1, 1])
    with main_left_col:
        st.header("Benchmarks MLP (CUDA vs PyTorch)")
        st.markdown(
            """
                Cette section compare les temps d'inférence d'un MLP implémenté :

                - en CUDA (`cuda_mlp`)
                - en PyTorch (`bench_mlp_pytorch.py`)
            """
        )
    with main_right_col:
        st.image("./images/mlp.png", caption="MLP Architecture", width=200)

    if "mlp_df" not in st.session_state:
        st.session_state.mlp_df = None
    if "mlp_ran" not in st.session_state:
        st.session_state.mlp_ran = False
    if "mlp_running" not in st.session_state:
        st.session_state.mlp_running = False

    if not st.session_state.mlp_ran and not st.session_state.mlp_running:
        st.session_state.mlp_running = True
        with st.spinner("Exécution automatique des benchmarks MLP..."):
            cuda_out = run_command(["./build/cuda_mlp"])
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
                    st.session_state.mlp_df = df
            else:
                st.warning("Une ou plusieurs exécutions MLP ont échoué ou ont renvoyé une sortie vide.")
    st.session_state.mlp_running = False
    st.session_state.mlp_ran = True



    df = st.session_state.mlp_df
    if df is not None and not df.empty:
        left_col, right_col = st.columns([1, 1])
        df_display = df.drop(columns=["input_dim", "output_dim", "speedup (torch/cuda)"], errors='ignore')
        df_display = df_display.sort_values(["hidden_dim", "batch_size"]).reset_index(drop=True)

        with left_col:
            st.subheader("Tableau comparatif")
            st.dataframe(df_display)
        with right_col:
            vis_col, batch_col = st.columns([4, 1])
            with vis_col:
                st.subheader("Visualisations")

            with batch_col:
                hidden_values = sorted(df["hidden_dim"].dropna().unique())

                index_hidden = 0
                if "visual_hidden" in st.session_state:
                    try:
                        index_hidden = hidden_values.index(st.session_state.visual_hidden)
                    except ValueError:
                        pass

                selected_hidden = st.selectbox(
                    "hidden dim",
                    hidden_values,
                    index=index_hidden,
                    key="visual_hidden",
                )


            st.markdown("---")
            df_h = df[df["hidden_dim"] == selected_hidden].copy()
            if not df_h.empty:
                df_h_group = (
                    df_h.groupby("batch_size")[["time_ms_cuda", "time_ms_torch"]]
                    .mean()
                    .sort_index()
                )
                st.line_chart(df_h_group)
            else:
                st.write("Aucune donnée pour le hidden_dim sélectionné.")

    else:
        st.info("Les benchmarks MLP ont été lancés automatiquement — aucun résultat exploitable n'a été retourné.")


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


def run_cnn_benchmarks_once():
    """
    Lance automatiquement le benchmark CNN une seule fois par session.
    Retourne (t_cuda, t_torch) ou (None, None) en cas d'erreur.
    """
    if "cnn_ran" not in st.session_state:
        st.session_state.cnn_ran = False
    if "cnn_running" not in st.session_state:
        st.session_state.cnn_running = False

    if not st.session_state.cnn_ran and not st.session_state.cnn_running:
        st.session_state.cnn_running = True
        with st.spinner("Exécution automatique du benchmark CNN..."):
            cuda_out = run_command(["./src/cnn"])
            torch_out = run_command([sys.executable, "./bench_cnn_pytorch.py"])
            t_cuda = None
            t_torch = None
            if cuda_out and torch_out:
                t_cuda = parse_single_time_line(cuda_out, "Temps moyen d'inférence CNN")
                t_torch = parse_single_time_line(torch_out, "Temps moyen d'inférence CNN PyTorch")
                if t_cuda is None or t_torch is None:
                    st.warning("Impossible de parser les temps CNN (format inattendu).")
            else:
                st.warning("Une ou plusieurs exécutions CNN ont échoué ou renvoyé une sortie vide.")
        st.session_state.cnn_running = False
        st.session_state.cnn_ran = True
        st.session_state.cnn_times = (t_cuda, t_torch)
    return st.session_state.get("cnn_times", (None, None))


def section_cnn():
    st.header("Benchmark CNN (CUDA vs PyTorch)")

    st.markdown(
        """
Cette section lance :

- `cuda_cnn` (implémentation CUDA du CNN)
- `bench_cnn_pytorch.py` (implémentation PyTorch équivalente)
        """
    )

    t_cuda, t_torch = run_cnn_benchmarks_once()

    if t_cuda is None or t_torch is None:
        st.info("Les benchmarks CNN ont été lancés automatiquement — résultats non disponibles ou parsing échoué.")
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


def run_rnn_benchmarks_once():
    """
    Lance automatiquement le benchmark RNN une seule fois par session.
    Retourne (t_cuda, t_torch) ou (None, None) en cas d'erreur.
    """
    if "rnn_ran" not in st.session_state:
        st.session_state.rnn_ran = False
    if "rnn_running" not in st.session_state:
        st.session_state.rnn_running = False

    if not st.session_state.rnn_ran and not st.session_state.rnn_running:
        st.session_state.rnn_running = True
        with st.spinner("Exécution automatique du benchmark RNN..."):
            cuda_out = run_command(["./src/rnn"])
            torch_out = run_command([sys.executable, "./bench_rnn_pytorch.py"])
            t_cuda = None
            t_torch = None
            if cuda_out and torch_out:
                # On affiche les sorties brutes (utile pour debug) puis on parse
                st.subheader("Sortie brute CUDA")
                st.code(cuda_out)
                st.subheader("Sortie brute PyTorch")
                st.code(torch_out)

                t_cuda = parse_single_time_line(cuda_out, "Temps moyen d'inférence RNN")
                t_torch = parse_single_time_line(torch_out, "Temps moyen d'inférence RNN PyTorch")

                if t_cuda is None or t_torch is None:
                    st.warning("Impossible de parser les temps RNN (format inattendu).")
            else:
                st.warning("Une ou plusieurs exécutions RNN ont échoué ou renvoyé une sortie vide.")
        st.session_state.rnn_running = False
        st.session_state.rnn_ran = True
        st.session_state.rnn_times = (t_cuda, t_torch)
    return st.session_state.get("rnn_times", (None, None))


def section_rnn():
    st.header("Benchmark RNN (CUDA vs PyTorch)")

    t_cuda, t_torch = run_rnn_benchmarks_once()

    if t_cuda is None or t_torch is None:
        st.info("Les benchmarks RNN ont été lancés automatiquement — résultats non disponibles ou parsing échoué.")
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
