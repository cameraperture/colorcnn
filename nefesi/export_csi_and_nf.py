#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions.network_data2 import NetworkData

# ─────────── EDIT THESE PATHS ───────────
SAVE_PATH   = "/Users/charlotteimbert/Documents/nefesi_outputs/VGG16_subset_bright_-50"
OBJ_IN      = os.path.join(SAVE_PATH, "Normal_class.obj")  # <-- your actual file
LAYERS      = ["features.0","features.2","features.5","features.7"]
NF_GRID_DIR = os.path.join(SAVE_PATH, "nf_grids")
CSV_OUT     = os.path.join(SAVE_PATH, "csi_values.csv")
GRID_SIZE   = 4   # 4×4 montage = first 16 neurons per layer
# ─────────────────────────────────────────

def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Scale an RGB patch to [0,1]."""
    mn, mx = patch.min(), patch.max()
    return (patch - mn)/(mx - mn) if mx>mn else patch - mn

def main():
    os.makedirs(NF_GRID_DIR, exist_ok=True)

    # 1) Load the single .obj that has activations + NF + (saved) CSI
    nd = NetworkData.load_from_disk(OBJ_IN)

    records = []

    # 2) For each layer, make the NF grid and collect metrics
    for layer in LAYERS:
        n_neurons = nd.get_len_neurons_of_layer(layer)

        # a) 4×4 grid of the first 16 NF patches
        fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(8,8))
        for idx, ax in enumerate(axes.flat):
            ax.axis("off")
            if idx < n_neurons:
                neuron = nd.get_neuron_of_layer(layer, idx)
                patch  = neuron._neuron_feature
                ax.imshow(normalize_patch(patch))
        fig.suptitle(f"{layer} — first {min(GRID_SIZE**2, n_neurons)} neurons")
        out_png = os.path.join(NF_GRID_DIR,
                               f"{layer.replace('.','_')}_nf_grid.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print("Saved NF grid →", out_png)

        # b) Gather each neuron’s max_activation & CSI (call the method!)
        ld = nd.get_layer_by_name(layer)
        for i in range(n_neurons):
            neuron = nd.get_neuron_of_layer(layer, i)
            max_act = float(neuron.activations[0])
            # call the bound method to get a float
            csi_val = neuron.color_selectivity_idx_new(
                nd,            # the NetworkData
                ld,            # this layer’s LayerData
                nd.dataset     # the ImageDataset
            )
            records.append({
                "layer": layer,
                "neuron_idx": i,
                "max_activation": max_act,
                "csi":            csi_val
            })

    # 3) Dump CSV
    df = pd.DataFrame(records)
    df.to_csv(CSV_OUT, index=False)
    print(f"Wrote CSI+activation CSV → {CSV_OUT}")

if __name__=="__main__":
    main()
