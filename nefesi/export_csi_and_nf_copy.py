#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions.network_data2 import NetworkData

# ─── USER CONFIG ─────────────────────────────────────────────────────────────
SAVE_PATH = "/Users/charlotteimbert/Documents/nefesi_outputs/VGG16_subset_bright_50"
OBJ_FINAL = os.path.join(SAVE_PATH, "Normal_class.obj")    # your final .obj
NF_DIR    = os.path.join(SAVE_PATH, "nf_grids")
CSV_OUT   = os.path.join(SAVE_PATH, "metrics_all.csv")
LAYERS    = ["features.0", "features.2", "features.5", "features.7"]
# ─────────────────────────────────────────────────────────────────────────────

def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Scale an RGB patch to [0,1]."""
    mn, mx = patch.min(), patch.max()
    return (patch - mn) / (mx - mn) if mx > mn else patch - mn

def main():
    # prepare output dirs
    os.makedirs(NF_DIR, exist_ok=True)

    # load the fully-populated NetworkData object
    nd = NetworkData.load_from_disk(OBJ_FINAL)

    all_records = []

    for layer in LAYERS:
        n_neurons = nd.get_len_neurons_of_layer(layer)
        cols      = int(math.ceil(math.sqrt(n_neurons)))
        rows      = int(math.ceil(n_neurons / cols))

        # 1) Make the NF grid
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols*1.5, rows*1.5),
                                 squeeze=False)
        for idx in range(rows*cols):
            r, c = divmod(idx, cols)
            ax   = axes[r][c]
            ax.axis("off")
            if idx < n_neurons:
                neuron = nd.get_neuron_of_layer(layer, idx)
                patch  = neuron._neuron_feature
                ax.imshow(normalize_patch(patch))
                ax.set_title(f"{idx}", fontsize=6)

        fig.suptitle(f"{layer} — all {n_neurons} neurons", fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.95])
        out_png = os.path.join(NF_DIR, f"{layer.replace('.','_')}_all_nfs.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print("Saved full NF grid →", out_png)

        # 2) Compute & collect metrics
        ld = nd.get_layer_by_name(layer)
        for idx in range(n_neurons):
            neuron = nd.get_neuron_of_layer(layer, idx)
            max_act = float(neuron.activations[0])
            # **call** the method to get a float
            csi_val = neuron.color_selectivity_idx_new(nd, ld, nd.dataset)
            all_records.append({
                "layer": layer,
                "neuron_idx": idx,
                "max_activation": max_act,
                "csi":            csi_val
            })

    # 3) Write out CSV
    df = pd.DataFrame(all_records)
    df.to_csv(CSV_OUT, index=False)
    print(f"Wrote metrics CSV → {CSV_OUT} ({len(df)} rows)")

if __name__ == "__main__":
    main()
