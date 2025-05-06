
            SAVE_PATH,
            f"imagenet64_decoder_features.{idx}.obj"
        )
        with open(dump_path, 'rb') as f:
            nf_arr = pickle.load(f, fix_imports=True, encoding='latin1')
        ld = Nefesimodel.get_layer_by_name(layer_name)
        ld.neuron_features = nf_arr
        print(f"  Attached NF array for {layer_name}, shape {nf_arr.shape}")

    Nefesimodel.save_to_disk('withNF')
    print("Saved object with NF included")

    # 7) Compute CSI for every neuron
    print("Computing CSI for each neuron…")
    for layer in Nefesimodel.get_layers_name():
        ld = Nefesimodel.get_layer_by_name(layer)
        print(" Layer:", layer)
        for idx in range(Nefesimodel.get_len_neurons_of_layer(layer)):
            neuron = Nefesimodel.get_neuron_of_layer(layer, idx)
            csi_val = get_color_selectivity_index_new(
                neuron_data = neuron,
                model       = Nefesimodel.model,
                layer_data  = ld,
                dataset     = dataset
            )
            neuron.color_selectivity_idx_new = csi_val

    Nefesimodel.save_to_disk('Normal_class')
    print("Saved object with CSI included")

if __name__ == '__main__':
    main()
