# colorcnn

Hello! This is the repository for our (Charlotte and Yufei's) final project for NEUR189B - Artificial and Biological Vision.

Source code for this project was accessed at the Nefesi GitHub repository (https://github.com/CVC-CIC/nefesi) by GitHub user arias2202, which contains Python scripts designed to replicate Rafegas & Vanrell's 2018 paper, 'Color encoding in biologically-inspired CNNs'. Nefesi provides functions to calculate neuron features, activations, and color selectivity indices. These scripts had to be modified slightly as they were written originally in Python 2 and had some other minor bugs as well.

Color transformation functions were created to change image brightness and transformations, looping through all the training data folders. To do this, the change_brightness.py and change_contrast.py scripts were used from the Python/digital_image_processing GitHub repository (https://github.com/TheAlgorithms/Python/tree/master/digital_image_processing) by GitHub users MaximSmolskiy and cclauss. The script to do the transformations is provided here for reference (train_transformations.py). The transformed and original images are already provided in zip files in this repo.

The specific Python scripts to run for this project are in the nefesi folder. The scripts are called:
1_create_nefesi_model.py and 2_analyze_nefesi_model.py, and are executed in that order.

The first script calculates the activations, neuron features (NFs), and color selectivity index (CSI) for every neuron in the first 2 convolutional layers of VGG-16, pre-trained on ImageNet. The second script produces a grid to visualize the NFs, and exports the CSI and maximum activations for each neuron to a .csv file. The image file path needs to be specified each time the script is run (it does not loop through all the image folders) - so a csv is created for each transformation. These csv files were manually combined into 2 Excel spreadsheets (metrics_brightness and metrics_contrast) which are available in this repo. Figures were created from these .xlsx files using the .Rmd script also in this repo.
