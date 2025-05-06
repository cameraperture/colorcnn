import torch
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
from functions.network_data2 import NetworkData
from functions.image import ImageDataset
import interface_DeepFramework.DeepFramework as DeepF

def preproces_imagenet_img(imgs_hr):
    img = np.array(imgs_hr)
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406],
                           std =[0.229,0.224,0.225])
    ])
    return [transform(img)]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.to(device).eval()
    for n, m in model.named_modules():
        m.auto_name = n

    deepmodel = DeepF.deep_model(model)

    layers_interest = [
      ['features.0', 0],
      ['features.2', 0],
      ['features.5', 0],
      ['features.7', 0]
    ]
    Path_images = "/Users/charlotteimbert/Documents/tiny-imagenet-200/train_transformed/train_bright_50_subset/"

    dataset = ImageDataset(
      src_dataset=Path_images,
      target_size=(224,224),
      preprocessing_function=preproces_imagenet_img,
      color_mode='rgb'
    )

    save_path = "/Users/charlotteimbert/Documents/nefesi_outputs/VGG16_subset_bright_50/"

    Nefesimodel = NetworkData(
      model=deepmodel,
      layer_data=layers_interest,
      save_path=save_path,
      dataset=dataset,
       default_file_name='VGG16_subset_bright_50',
      input_shape=[(1,3,224,224)]
    )
    Nefesimodel.generate_neuron_data()

    # 1) Activations
    Nefesimodel.eval_network()
    print('Activation Calculus done!')
    Nefesimodel.save_to_disk('activations')

    # 2) Neuron Features
    Nefesimodel.calculateNF()
    print('NF done!')
    Nefesimodel.save_to_disk('withNF')
    print('Saved object with NF included')

    # 3) Color selectivity index
    dataset = Nefesimodel.dataset
    for layer in Nefesimodel.get_layers_name():
        layer_data = Nefesimodel.get_layer_by_name(layer)
        print(layer)
        for n in range(Nefesimodel.get_len_neurons_of_layer(layer)):
            neurona = Nefesimodel.get_neuron_of_layer(layer, n)
            neurona.color_selectivity_idx_new(
                Nefesimodel, layer_data, dataset
            )
    Nefesimodel.save_to_disk('Normal_class')

if __name__ == '__main__':
    main()
