# colorcnn

Hello! This is the repository for our (Charlotte and Yufei's) final project for NEUR189B - Artificial and Biological Vision.
The Python scripts to run are called:
create_nefesi_model.py and 2_analyze_nefesi_model.py, in that order.

The first script calculates the activations, neuron features (NFs), and color selectivity index (CSI) for every neuron in the first 2 convolutional layers of VGG-16, pre-trained on ImageNet. The second script produces a grid to visualize the NFs, and exports the CSI and maximum activations for each neuron to a .csv file.
