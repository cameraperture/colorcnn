# colorcnn

Hello! This is the repository for our (Charlotte and Yufei's) final project for NEUR189B - Artificial and Biological Vision.

Code for this project was taken from the Nefesi GitHub repository (https://github.com/CVC-CIC/nefesi) by GitHub user arias2202, which was designed to replicate Rafegas & Vanrell's 2018 paper, 'Color encoding in biologically-inspired CNNs'. Nefesi provides functions to calculate neuron features, activations, and color selectivity indices. These scripts had to be modified slightly as they were written originally in Python 2.

Color transformation functions were created to change image brightness and transformations. To do this, the change_brightness.py and change_contrast.py scripts were used from the Python/digital_image_processing GitHub repository (https://github.com/TheAlgorithms/Python/tree/master/digital_image_processing) by GitHub users MaximSmolskiy and cclauss.

The specific Python scripts to run for this project are called:
1_create_nefesi_model.py and 2_analyze_nefesi_model.py, in that order.

The first script calculates the activations, neuron features (NFs), and color selectivity index (CSI) for every neuron in the first 2 convolutional layers of VGG-16, pre-trained on ImageNet. The second script produces a grid to visualize the NFs, and exports the CSI and maximum activations for each neuron to a .csv file. The image file path needs to be specified each time the script is run (it does not loop through all the image folders) - so a csv is created for each transformation. These csv files were manually combined into 2 Excel spreadsheets (metrics_brightness and metrics_contrast) which are available in this repository. Figures were created from these .xlsx files using the .Rmd script also in this repo.
