#  HyperGator access and PyTorch data IO

The objectives of this homework are:
1. Practice accessing the HiperGator shared data folder.
2. Practice using PyTorch datasets and data loaders.

### Move your data to HiperGator (10 pts)
Create a folder  in the **shared** blue folder at `/blue/isc5935/shared/teamXX/data`.
The name should be in the format of `teamXX/data` where `XX` is the number of your team (1,2 or 3).

Copy the data of your final project to this folder in HiperGator cluster. 

Please describe the proposed organization of the data inside that folder.

Some examples of data organization are:
- `teamXX/data/raw/` - raw data
- `teamXX/data/processed/` - processed data
- `teamXX/data/processed/train/` - training data
- `teamXX/data/processed/test/` - test data
- `teamXX/data/processed/val/` - validation data
- `teamXX/data/processed/train/images/` - training images
- `teamXX/data/processed/train/labels/` - training labels
- `teamXX/src/` - source code

or you can use any other organization that you think is appropriate, just describe it in your *answer* notebook. 


### Dataset (10 pts)
Create a PyTorch [custom dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) **MyGenerator.py** that loads the data from the folder or folders you created in the previous step.
One of the input parameters of your constructor should be *tranform*, which can be a transformation
applied to each of your generated samples.

The dataset must have at least three methods:
- `__len__` - returns the number of samples in the dataset
- `__getitem__` - returns a sample from the dataset (input and target)
- `__init__` - initializes the dataset

### Data loader and visualizer (10 pts)
In a jupyter notebook, create a simple ipwidget to visualize the data from your dataset.
In this widget you should be able to select the **batch size**. 
Using a button, the widget should display the images and corresponding target generated from the dataset for the selected batch size.

A simple example is available at [Datasets and DataLoader Example](https://github.com/olmozavala/ISC_5935_EXamples/blob/main/PyTorch/3_Data_Loaders.py).
