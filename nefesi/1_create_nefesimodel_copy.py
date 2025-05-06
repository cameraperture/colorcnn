import torch
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import functools
from functions.network_data2 import NetworkData
import types
BATCH_SIZE = 100
from  functions.image import ImageDataset
import interface_DeepFramework.DeepFramework as DeepF

def preproces_imagenet_img( imgs_hr):

    img=np.array(imgs_hr)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    tnsr = [transform(img)]


    return tnsr




def main():

    folder_dir = "/Users/charlotteimbert/nefesi/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.to(device).eval()


    for n, m in model.named_modules():
        m.auto_name = n
        print(n)

    deepmodel = DeepF.deep_model(model)


    # Create a list with the layers that you want to analyze and 0 if they are encoding or 1 if they are decoding
    # layers_interest = [['features.1', 0], ['features.3', 0], ['features.6', 0], ['features.8', 0]]
    layers_interest = [['features.0', 0], ['features.2', 0], ['features.5', 0], ['features.7', 0] ]

    Path_images = "/Users/charlotteimbert/Documents/tiny-imagenet-200/train_subset/"

    preproces_function=preproces_imagenet_img
    dataset = ImageDataset(
        src_dataset=Path_images,target_size=(224,224), preprocessing_function=preproces_function,color_mode='rgb')
    print(">>> Path_images =", Path_images)
    print(">> dataset.src_dataset =", dataset.src_dataset)

    # Path where you will save your results
    save_path= "/Users/charlotteimbert/Documents/nefesi_outputs/VGG16_subset/"



    Nefesimodel= NetworkData(model=deepmodel,layer_data=layers_interest,save_path = save_path, dataset=dataset,default_file_name = 'VGG16_subset',input_shape=[(1,3,224,224)])
    Nefesimodel.dataset = dataset
    Nefesimodel.generate_neuron_data()

    # calculate the top scoring images
    Nefesimodel.eval_network()

    print('Activation Calculus done!')
    Nefesimodel.save_to_disk('activations')

    # Nefesimodel=NetworkData.load_from_disk("Model_generation\Nefesi_models\VGG16\VGGPartialSave100WithoutNF.obj")

    # calculate the Neuron feature of each neuron (weighted average of top scoring images)
    Nefesimodel.calculateNF()
    print('NF done!')


    # # calculate the Color selectivity of each neuron
    # dataset = Nefesimodel.dataset
    # for layer in Nefesimodel.get_layers_name():
    #     layer_data = Nefesimodel.get_layer_by_name(layer)
    #     print(layer)
    #     for n in range(Nefesimodel.get_len_neurons_of_layer(layer)):
    #         neurona = Nefesimodel.get_neuron_of_layer(layer, n)
    #         neurona.color_selectivity_idx_new(Nefesimodel, layer_data, dataset)
    # Nefesimodel.save_to_disk('Normal_class')
    # #
    # # # calculate the Similarity Index of each neuron in the same layer
    # # for layer in Nefesimodel.get_layers_name():
    # #     print(layer)
    # #     Nefesimodel.get_layer_by_name(layer).similarity_index = None
    # #     x=Nefesimodel.similarity_idx(layer)
    # #     print(x)
    # #
    # # Nefesimodel.save_to_disk('similarity')
    #






if __name__ == '__main__':
    main()
