# Malaria Detection using Pytorch
For this project blood samples with and without parasites have been considered. The overall goal is to make sure we are able to classify whether the person has Malaria or not based on the blood image samples.

## Model Structure
The project involved using of pretrained model architecture from timm library which is associated with Pytorch. The pretrained model which was used is the Vision Transformer. 
The model was able to provide predictions with F1 score of 0.808. 
The number of epochs trained is 10.

![0 YRDqyaLnCJscrYWV](https://github.com/hemanthh17/malaria-detection-torch-docker/assets/49975886/81303bef-61ec-40df-ba5c-314d6c065cdb)


## Process Involved
The dataset is from [Kaggle](https://www.kaggle.com/datasets/nipunarora8/malaria-detection-dataset). The data was preprocessed and resized to uniformity to (224,224), all of the training and other parameters can be found in the scripts/config.py file.
Post training, the model was saved, and a seprate script to define the model and the function call to classify the given image was initialised. The further step is to design a Flask app in order to locally host the application. The results are displayed in a new web page.

## Dockerizing
In order to keep the dependencies uniform, the entire environment was dockerized. The image of this can be found in the [Docker Hub](https://hub.docker.com/r/hemanthh17/torchmalaria). 
```
docker pull docker pull hemanthh17/torchmalaria:latest
```
To use this application the image can be pulled and a new container cna be created at the destination to run the application. **(Not yet uploaded)**
```
docker run --name malariadetection -p 5000 hemanthh17/torchmalaria:latest
```
If you want to build the image from scratch
```
docker build -t malariatorch .
```
