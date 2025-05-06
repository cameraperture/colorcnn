import torch
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
from functions.network_data2 import NetworkData
from functions.image import ImageDataset
import interface_DeepFramework.DeepFramework as DeepF

# Script from 1_create_nefesimodel.py in the nefesi GitHub repo, with modifications for it to run locally and using VGG16 instead of UNet

# convert imgs to tensor
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
    model  = vgg16(weights=VGG16_Weights.IMAGENET1K_V1) #use vgg16 pretrained model
    model.to(device).eval() #move the model to the appropriate device and set it to evaluation mode
    for n, m in model.named_modules(): 
        # n: the string name of the module (eg.'features.3')
        # m: the actual module object (eg. a Conv2d, ReLU, or Linear layer)
        m.auto_name = n #assign an auto_name attribute to name the object

    deepmodel = DeepF.deep_model(model) #change the model type to deep_model in the nefesi class for later steps

    layers_interest = [
      ['features.0', 0],  #conv1_1
      ['features.2', 0],  #conv1_2
      ['features.5', 0],  #conv2_1
      ['features.7', 0]   #conv2_2
    ]
    Path_images = "/Users/charlotteimbert/Documents/tiny-imagenet-200/train_transformed/train_contrast_100_subset/"

    dataset = ImageDataset(
      src_dataset=Path_images,
      target_size=(224,224), #change to imagenet img size
      preprocessing_function=preproces_imagenet_img,
      color_mode='rgb'
    )

    save_path = "/Users/charlotteimbert/Documents/nefesi_outputs/VGG16_subset_contrast_100/"

    Nefesimodel = NetworkData(
      model=deepmodel,
      layer_data=layers_interest,
      save_path=save_path,
      dataset=dataset,
       default_file_name='VGG16_subset_contrast_100',
      input_shape=[(1,3,224,224)] #batch size, channel number, H, W
    )
    Nefesimodel.generate_neuron_data()

    #1) Get neuron activations from the model
    Nefesimodel.eval_network()
    print('Activation Calculus done!')
    Nefesimodel.save_to_disk('activations')

    #2) Get neuron Features
    Nefesimodel.calculateNF()
    print('NF done!')
    Nefesimodel.save_to_disk('withNF')
    print('Saved object with NF included')

    #3) Color selectivity index
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
