# MD ML
End-to-end implementation of Malaria detection using advanced highly-interpretable and robust prototype-based ML model Image Generalized Matrix Learning Vector Quantization as well as deep convolutional neural network model built on the transfer learning architecture of InceptionResNetV2 with some custom adjustments.

## What is it?
MD ML is a tool for detecting malaria based on the parasites class and uninfected class.



## Prototype-based 

The malaria prototypes were trained with the ```ImageGMLVQ``` prototype-based models. The prototype initialization was done uniformly on one prototype per class basis. Predictions for each malaria test case is based on the trained protypes displayed below.


##### Initialised prototypes for  parasite class and uninfected class after preprocessing transformation

![download](https://user-images.githubusercontent.com/82911284/175720641-109baf6d-653f-435d-8498-bde91a36ab7a.png)


### ImageGMLVQ model
Trained  Prototypes of parasites class and uninfected class from ImageGMLVQ model

![download](https://user-images.githubusercontent.com/82911284/175665381-fb6b1c5a-146b-4e6e-a647-a006e15dff00.png)


## Non-Prototype-based

A deep convolutional neural network model built on transfer learning with ```InceptionResNetV2``` architecture with some custom adjustments was used to train the malaria cell image data for the parasite and uninfected class.




