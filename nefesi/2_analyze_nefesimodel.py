
#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from functions.network_data2 import NetworkData

def normalize(patch: np.ndarray) -> np.ndarray:
    """Scale to [0,1] for display."""
    mn, mx = patch.min(), patch.max()
    return (patch - mn) / (mx - mn) if mx > mn else patch - mn

def main():
    nd = NetworkData.load_from_disk(
        "/Users/charlotteimbert/Documents/nefesi_outputs/VGG16_subset/Normal_class.obj"
    )

    layers = ['features.0', 'features.2', 'features.5', 'features.7']
    num_to_show = 10
    cols = 5
    rows = 2

    for layer in layers:
        # create one figure per layer
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axes = axes.flat  # flatten for easy iteration

        for i in range(num_to_show):
            ax = axes[i]
            neuron = nd.get_neuron_of_layer(layer, i)
            patch  = neuron._neuron_feature

            # normalize and plot
            ax.imshow(normalize(patch))
            ax.set_title(f"#{i}", fontsize=8)
            ax.axis('off')

        # any leftover axes hide
        for ax in axes[num_to_show:]:
            ax.axis('off')

        plt.suptitle(f"{layer} — first {num_to_show} neurons", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # save or show
        out_png = f"{layer.replace('.','_')}_nf_grid.png"
        fig.savefig(out_png, dpi=150)
        print(f"Saved grid → {out_png}")
        plt.close(fig)

if __name__ == "__main__":
    main()
