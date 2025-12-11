import re
import subprocess
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import json

def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        new_col = col
        if col == "input_dim":
            new_col = "Input dim"
        elif col == "hidden_dim":
            new_col = "Hidden dim"
        elif col == "output_dim":
            new_col = "Output dim"
        elif col == "batch_size":
            new_col = "Batch size"
        elif col == "seq_len":
            new_col = "Seq. length"
        elif col == "N":
            new_col = "Batch size (N)"
        elif col == "Cin":
            new_col = "Input channels (Cin)"
        elif col == "Cout":
            new_col = "Output channels (Cout)"
        elif col == "HxW":
            new_col = "Image size (HxW)"
        elif col == "K":
            new_col = "Kernel size (K)"
        elif col == "FC":
            new_col = "FC out"
        elif col.startswith("time_ms_") and not (
            col.startswith("time_ms_total_")
            or col.startswith("time_ms_gpu_only_")
            or col.startswith("time_ms_copies_")
        ):
            impl = col.replace("time_ms_", "")
            impl_pretty = impl.upper() if impl.lower().startswith("cuda") else impl.capitalize()
            new_col = f"Time {impl_pretty} (ms)"
        elif col.startswith("time_ms_total_"):
            impl = col.replace("time_ms_total_", "")
            impl_pretty = impl.upper() if impl.lower().startswith("cuda") else impl.capitalize()
            new_col = f"Time {impl_pretty} total (ms)"
        elif col.startswith("time_ms_gpu_only_"):
            impl = col.replace("time_ms_gpu_only_", "")
            impl_pretty = impl.upper() if impl.lower().startswith("cuda") else impl.capitalize()
            new_col = f"Time {impl_pretty} GPU-only (ms)"
        elif col.startswith("time_ms_copies_"):
            impl = col.replace("time_ms_copies_", "")
            impl_pretty = impl.upper() if impl.lower().startswith("cuda") else impl.capitalize()
            new_col = f"Copies latency {impl_pretty} (ms)"
        if new_col != col:
            rename_map[col] = new_col
    return df.rename(columns=rename_map)


def run_command(cmd, timeout=300):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout

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
        st.header("Benchmarks MLP")
        st.markdown(
            """
                Cette section compare les temps d'inférence d'un MLP implémenté entre :

                - CUDA
                - PyTorch
            """
        )
    with main_right_col:
        st.image("./images/mlp.png", caption="MLP Architecture", width=800)

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
            df_cuda = parse_mlp_lines(cuda_out, "cuda")
            df_torch = parse_mlp_lines(torch_out, "torch")
            df = pd.merge(
                df_cuda,
                df_torch,
                on=["input_dim", "hidden_dim", "output_dim", "batch_size"],
                how="outer",
            )
            st.session_state.mlp_df = df
    st.session_state.mlp_running = False
    st.session_state.mlp_ran = True
    df = st.session_state.mlp_df

    left_col, right_col = st.columns([1, 1])
    df_display = df.drop(columns=["input_dim", "output_dim", "speedup (torch/cuda)"], errors='ignore')
    df_display = df_display.sort_values(["hidden_dim", "batch_size"]).reset_index(drop=True)
    df_display = prettify_columns(df_display)

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
                index_hidden = hidden_values.index(st.session_state.visual_hidden)
            selected_hidden = st.selectbox(
                "hidden dim",
                hidden_values,
                index=index_hidden,
                key="visual_hidden",
            )

        df_h = df[df["hidden_dim"] == selected_hidden].copy()
        df_h_group = (
            df_h.groupby("batch_size")[["time_ms_cuda", "time_ms_torch"]]
            .mean()
            .sort_index()
        )
        df_h_group = prettify_columns(df_h_group)
        st.line_chart(df_h_group)


def load_cnn_json(filename: str, label: str) -> pd.DataFrame:
    path = Path(filename)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows_map = {}
    for item in data:
        cfg_str = item.get("configuration", "")
        if not cfg_str:
            continue
        parts = [p.strip() for p in cfg_str.split(",")]
        cfg_dict = {}
        for p in parts:
            if "=" not in p:
                continue
            k, v = [x.strip() for x in p.split("=", 1)]
            cfg_dict[k] = v
        N = int(cfg_dict["N"])
        Cin = int(cfg_dict["Cin"])
        HxW = cfg_dict["HxW"]
        Cout = int(cfg_dict["Cout"])
        K = int(cfg_dict["K"])
        FC = int(cfg_dict["FC"])
        key = (N, Cin, HxW, Cout, K, FC)
        if key not in rows_map:
            rows_map[key] = {
                "N": N,
                "Cin": Cin,
                "HxW": HxW,
                "Cout": Cout,
                "K": K,
                "FC": FC,
            }
        row = rows_map[key]
        mode = item.get("mode", "").strip().lower()
        impl_suffix = f"{label}_{mode}" if mode else label
        total = float(item["temps_total_ms"])
        gpu_only = float(item["temps_gpu_only_ms"])
        row[f"time_ms_total_{impl_suffix}"] = total
        row[f"time_ms_gpu_only_{impl_suffix}"] = gpu_only
    return pd.DataFrame(list(rows_map.values()))


def section_cnn():
    st.header("Benchmarks CNN")
    main_left_col, main_right_col = st.columns([1, 1])
    with main_left_col:
        st.markdown(
            """
                Cette section compare les temps d'inférence d'un CNN implémenté :

                - CUDA (**NAIVE** et **TILED**)
                - PyTorch
            """
        )
    with main_right_col:
        img_path = Path("./images/cnn.png")
        if img_path.exists():
            st.image(str(img_path), caption="CNN Architecture", width=800)

    if "cnn_df" not in st.session_state:
        st.session_state.cnn_df = None
    if "cnn_ran" not in st.session_state:
        st.session_state.cnn_ran = False
    if "cnn_running" not in st.session_state:
        st.session_state.cnn_running = False

    if not st.session_state.cnn_ran and not st.session_state.cnn_running:
        st.session_state.cnn_running = True
        with st.spinner("Chargement des résultats CNN depuis les JSON..."):
            dfs = []

            # 1) PyTorch
            try:
                df_pytorch = load_cnn_json("results/benchmark_cnn_pytorch.json", label="pytorch")
                if not df_pytorch.empty:
                    dfs.append(df_pytorch)
            except FileNotFoundError:
                st.warning("Fichier 'results/benchmark_cnn_pytorch.json' introuvable (résultats PyTorch).")
            except Exception as e:
                st.warning(f"Erreur lors du chargement de 'results/benchmark_cnn_pytorch.json': {e}")

            # 2) CUDA (NAIVE / TILED)
            try:
                df_cuda = load_cnn_json("results/benchmark_cnn.json", label="cuda")
                if not df_cuda.empty:
                    dfs.append(df_cuda)
            except FileNotFoundError:
                st.warning("Fichier 'results/benchmark_cnn.json' introuvable (résultats CUDA).")
            except Exception as e:
                st.warning(f"Erreur lors du chargement de 'results/benchmark_cnn.json': {e}")

            if not dfs:
                st.warning("Aucun résultat CNN n'a pu être chargé depuis les JSON.")
                st.session_state.cnn_df = None
            else:
                df_merged = dfs[0]
                for d in dfs[1:]:
                    df_merged = pd.merge(
                        df_merged,
                        d,
                        on=["N", "Cin", "HxW", "Cout", "K", "FC"],
                        how="outer",
                    )
                st.session_state.cnn_df = df_merged

        st.session_state.cnn_running = False
        st.session_state.cnn_ran = True

    df = st.session_state.cnn_df

    if df is None or df.empty:
        st.info("Aucun résultat CNN disponible pour affichage.")
        return

    # --- Tableau brut ---
    st.subheader("Tableau comparatif")
    df_display = df.sort_values(["N", "Cin", "HxW", "Cout", "K", "FC"]).reset_index(drop=True)
    df_display = prettify_columns(df_display)
    st.dataframe(df_display)

    col_hxw, col_cout, col_k, col_fc, _ = st.columns(5)
    current_df = df.copy()
    filters = {}
    filters["Cin"] = 3
    current_df = current_df[current_df["Cin"] == 3]
    
    # HxW
    with col_hxw:
        vals_hxw = sorted(current_df["HxW"].dropna().unique())
        if len(vals_hxw) == 1:
            selected_hxw = vals_hxw[0]
            st.markdown(f"**HxW** : `{selected_hxw}`")
        else:
            selected_hxw = st.selectbox(
                "HxW",
                vals_hxw,
                key="cnn_filter_hxw",
            )
        filters["HxW"] = selected_hxw
        current_df = current_df[current_df["HxW"] == selected_hxw]

    # Cout
    with col_cout:
        vals_cout = sorted(current_df["Cout"].dropna().unique())
        if len(vals_cout) == 1:
            selected_cout = vals_cout[0]
            st.markdown(f"**Cout** : `{selected_cout}`")
        else:
            selected_cout = st.selectbox(
                "Cout",
                vals_cout,
                key="cnn_filter_cout",
            )
        filters["Cout"] = selected_cout
        current_df = current_df[current_df["Cout"] == selected_cout]

    # K
    with col_k:
        vals_k = sorted(current_df["K"].dropna().unique())
        if len(vals_k) == 1:
            selected_k = vals_k[0]
            st.markdown(f"**K** : `{selected_k}`")
        else:
            selected_k = st.selectbox(
                "K",
                vals_k,
                key="cnn_filter_k",
            )
        filters["K"] = selected_k
        current_df = current_df[current_df["K"] == selected_k]

    # FC
    with col_fc:
        vals_fc = sorted(current_df["FC"].dropna().unique())
        if len(vals_fc) == 1:
            selected_fc = vals_fc[0]
            st.markdown(f"**FC** : `{selected_fc}`")
        else:
            selected_fc = st.selectbox(
                "FC",
                vals_fc,
                key="cnn_filter_fc",
            )
        filters["FC"] = selected_fc
        current_df = current_df[current_df["FC"] == selected_fc]

    # ========== FILTRAGE + COURBES ==========
    x_axis = "N"  # taille de batch sur l'axe X

    df_filtered = df.copy()
    for dim, val in filters.items():
        df_filtered = df_filtered[df_filtered[dim] == val]

    df_filtered = df_filtered.sort_values(x_axis)
    if not df_filtered.empty:
        df_filtered = df_filtered.set_index(x_axis)

    # Colonnes de temps TOTAL / GPU-only
    total_cols = [c for c in df_filtered.columns if c.startswith("time_ms_total_")]
    gpu_cols = [c for c in df_filtered.columns if c.startswith("time_ms_gpu_only_")]

    # On ne garde à l'affichage que les séries qui ont au moins 2 points
    valid_total_cols = [c for c in total_cols if df_filtered[c].count() >= 2]
    single_total_cols = [c for c in total_cols if df_filtered[c].count() == 1]

    valid_gpu_cols = [c for c in gpu_cols if df_filtered[c].count() >= 2]
    single_gpu_cols = [c for c in gpu_cols if df_filtered[c].count() == 1]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Temps TOTAL")
        if valid_total_cols:
            st.line_chart(prettify_columns(df_filtered[valid_total_cols]))
        else:
            st.info("Aucune série avec au moins deux valeurs de N à afficher (temps total).")

        if single_total_cols:
            st.caption(
                "Non affiché (une seule mesure de N) : "
                + ", ".join(single_total_cols)
            )

    with right:
        st.subheader("Temps GPU-ONLY")
        if valid_gpu_cols:
            st.line_chart(prettify_columns(df_filtered[valid_gpu_cols]))
        else:
            st.info("Aucune série avec au moins deux valeurs de N à afficher (GPU-only).")

        if single_gpu_cols:
            st.caption(
                "Non affiché (une seule mesure de N) : "
                + ", ".join(single_gpu_cols)
            )



def load_rnn_json(filename: str, label: str, schema: str) -> pd.DataFrame:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(filename)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.get("resultats", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []

    rows = []

    for item in items:
        # Dimensions (communes aux 3 fichiers)
        try:
            input_dim = int(item["D_input"])
            hidden_dim = int(item["H_hidden"])
            seq_len    = int(item["T_seq_len"])
            batch_size = int(item["N_batch"])
        except KeyError as e:
            raise KeyError(f"Clé de dimension manquante dans {filename} : {e}") from e

        row = {
            "input_dim":  input_dim,
            "hidden_dim": hidden_dim,
            "seq_len":    seq_len,
            "batch_size": batch_size,
        }

        # temps_total_ms : présent partout
        try:
            total = float(item["temps_total_ms"])
        except KeyError as e:
            raise KeyError(f"Clé 'temps_total_ms' manquante dans {filename}") from e

        row[f"time_ms_total_{label}"] = total

        # temps_gpu_only_ms : présent pour rnn2 et pytorch
        if schema in ("rnn2", "pytorch"):
            gpu_only = item.get("temps_gpu_only_ms", None)
            if gpu_only is not None:
                row[f"time_ms_gpu_only_{label}"] = float(gpu_only)

            # latence_copies_ms éventuellement présente -> on la stocke si dispo
            copies = item.get("latence_copies_ms", None)
            if copies is not None:
                row[f"time_ms_copies_{label}"] = float(copies)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def section_rnn():
    st.header("Benchmarks RNN")
    main_left_col, _ = st.columns([1, 1])
    with main_left_col:
        st.markdown(
            """
            Cette section compare les temps d'inférence de deux RNN implémentés :

            - CUDA (premier rnn)
            - CUDA (second rnn plus optimisé)
            - PyTorch
            """
        )

    if "rnn_df" not in st.session_state:
        st.session_state.rnn_df = None
    if "rnn_ran" not in st.session_state:
        st.session_state.rnn_ran = False
    if "rnn_running" not in st.session_state:
        st.session_state.rnn_running = False

    # --- Chargement des JSON une seule fois par session ---
    if not st.session_state.rnn_ran and not st.session_state.rnn_running:
        st.session_state.rnn_running = True
        with st.spinner("Chargement des résultats RNN depuis les JSON..."):
            dfs = []

            # Adapte les chemins/nom de fichiers ici selon ton arborescence
            # 1) rnn : temps_moyen_ms (schéma 'rnn')
            try:
                df_rnn = load_rnn_json("results/benchmark_rnn.json", label="cuda", schema="rnn")
                if not df_rnn.empty:
                    dfs.append(df_rnn)
            except FileNotFoundError:
                st.warning("Fichier 'results/benchmark_rnn.json' introuvable (résultats rnn).")
            except Exception as e:
                st.warning(f"Erreur lors du chargement de 'results/benchmark_rnn.json': {e}")

            # 2) rnn2 : temps_total_ms / temps_gpu_only_ms / latence_copies_ms (schéma 'rnn2')
            try:
                df_rnn2 = load_rnn_json("results/benchmark_rnn2.json", label="cuda2", schema="rnn2")
                if not df_rnn2.empty:
                    dfs.append(df_rnn2)
            except FileNotFoundError:
                st.warning("Fichier 'results/benchmark_rnn2.json' introuvable (résultats rnn2).")
            except Exception as e:
                st.warning(f"Erreur lors du chargement de 'results/benchmark_rnn2.json': {e}")

            # 3) PyTorch : array racine + Time_Total_ms... (schéma 'pytorch')
            try:
                df_torch = load_rnn_json(
                    "results/benchmark_rnn_pytorch.json",
                    label="torch",
                    schema="pytorch",
                )
                if not df_torch.empty:
                    dfs.append(df_torch)
            except FileNotFoundError:
                st.warning("Fichier 'results/benchmark_rnn_pytorch.json' introuvable (résultats PyTorch).")
            except Exception as e:
                st.warning(f"Erreur lors du chargement de 'results/benchmark_rnn_pytorch.json': {e}")

            if not dfs:
                st.warning("Aucun résultat RNN n'a pu être chargé depuis les JSON.")
                st.session_state.rnn_df = None
            else:
                df_merged = dfs[0]
                for d in dfs[1:]:
                    df_merged = pd.merge(
                        df_merged,
                        d,
                        on=["input_dim", "hidden_dim", "seq_len", "batch_size"],
                        how="outer",
                    )
                st.session_state.rnn_df = df_merged

        st.session_state.rnn_running = False
        st.session_state.rnn_ran = True

    df = st.session_state.rnn_df

    if df is None or df.empty:
        st.info("Aucun résultat RNN disponible pour affichage.")
        return

    st.subheader("Tableau comparatif")
    df_display = df.sort_values(
        ["input_dim", "hidden_dim", "seq_len", "batch_size"]
    ).reset_index(drop=True)
    df_display = prettify_columns(df_display)
    st.dataframe(df_display)

    col_hid, col_seq, _, _, _ = st.columns(5)
    current_df = df.copy()
    filters = {}
    filters["input_dim"] = 256
    current_df = current_df[current_df["input_dim"] == 256]

    # hidden_dim
    with col_hid:
        vals_hid = sorted(current_df["hidden_dim"].dropna().unique())
        if len(vals_hid) == 1:
            selected_hid = vals_hid[0]
            st.markdown(f"**hidden_dim**: `{selected_hid}`")
        else:
            selected_hid = st.selectbox(
                "hidden_dim",
                vals_hid,
                key="rnn_filter_hidden_dim",
            )
        filters["hidden_dim"] = selected_hid
        current_df = current_df[current_df["hidden_dim"] == selected_hid]

    # seq_len
    with col_seq:
        vals_seq = sorted(current_df["seq_len"].dropna().unique())
        if len(vals_seq) == 1:
            selected_seq = vals_seq[0]
            st.markdown(f"**seq_len**: `{selected_seq}`")
        else:
            selected_seq = st.selectbox(
                "seq_len",
                vals_seq,
                key="rnn_filter_seq_len",
            )
        filters["seq_len"] = selected_seq
        current_df = current_df[current_df["seq_len"] == selected_seq]

    # ========== FILTRAGE + COURBES ==========
    x_axis = "batch_size"

    df_filtered = df.copy()
    for dim, val in filters.items():
        df_filtered = df_filtered[df_filtered[dim] == val]

    df_filtered = df_filtered.sort_values(x_axis)
    if not df_filtered.empty:
        df_filtered = df_filtered.set_index(x_axis)

    total_cols = []
    gpu_cols = []
    if not df_filtered.empty:
        total_cols = [c for c in df_filtered.columns if c.startswith("time_ms_total_")]
        gpu_cols = [c for c in df_filtered.columns if c.startswith("time_ms_gpu_only_")]

        df_filtered_total = (
            df_filtered[total_cols].dropna(how="any") if total_cols else df_filtered
        )
        df_filtered_gpu = (
            df_filtered[gpu_cols].dropna(how="any") if gpu_cols else df_filtered
        )
    else:
        df_filtered_total = df_filtered
        df_filtered_gpu = df_filtered

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Temps TOTAL")
        if not df_filtered_total.empty and total_cols:
            st.line_chart(prettify_columns(df_filtered_total[total_cols]))
    with col_right:
        st.subheader("Temps GPU-ONLY")
        if not df_filtered_gpu.empty and gpu_cols:
            st.line_chart(prettify_columns(df_filtered_gpu[gpu_cols]))
        else:
            st.info("Aucune donnée de temps GPU-only à afficher pour cette configuration.")

def main():
    st.set_page_config(
        page_title="Sujet 7 - MLOps / CUDA",
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
