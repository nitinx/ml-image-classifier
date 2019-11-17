# Deep Learning
## Project: Image Classifier

## Project Overview
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. 

## Project Highlights
The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [PyTorch](https://pytorch.org/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Code

This project contains six files:

- `Image_Classifier_Project.ipynb`: This is the main file where you will be performing your work on the project.
- `cat_to_name.json`: A dictionary for mapping integer encoded category labels to actual names of the flowers.
- `helper.py`: A Python file containing helper code that is run behind-the-scenes. Do not modify
- `workspace_utils.py`: A Python module which includes an iterator wrapper called `keep_awake` and a context manager called `active_session` that can be used to maintain an active session during long-running processes. The two functions are equivalent, so use whichever fits better in your code. 
- `train.py`: A Python file that will train a new network on a dataset and save the model as a checkpoint.
- `predict.py`: A Python file that uses a trained network to predict the class for an input image.


### Run

For Part #1, in a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
ipython notebook Image_Classifier_Project.ipynb
```  
or
```bash
jupyter notebook Image_Classifier_Project.ipynb
```

This will open the iPython Notebook software and project file in your browser.

For Part #2, in a terminal or command window, navigate to the top-level project directory (that contains this README) and run the following commands:

- Train a new network on a data set with `train.py`
-- Basic usage: `python train.py data_directory`
-- Prints out training loss, validation loss, and validation accuracy as the network trains
-- Options:
--- Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
--- Choose architecture: `python train.py data_dir --arch "vgg13"`
--- Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
--- Use GPU for training: `python train.py data_dir --gpu`

- Predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.
-- Basic usage: `python predict.py /path/to/image checkpoint`
-- Options:
--- Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`
--- Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
--- Use GPU for inference: `python predict.py input checkpoint --gpu`

### Data

We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories. The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

