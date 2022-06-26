# MD ML
End-to-end implementation of Malaria detection using advanced highly-interpretable and robust prototype-based ML models (Image Generalized Learning Vector Quantization and  Image Generalized Matrix Learning Vector Quantization) as well as deep convolutional neural network model built on the transfer learning architecture of InceptionResNetV2 with some custom adjustments.

## What is it?
MD ML is a tool for detecting malaria based on the parasites class and uninfected class.

This project comes in handy for medical professionals who may want to leverage the stupendous capabilities of machine learning in Medical Intelligence Systems.

Expert machine learning practitioners who require a high level of control in terms of interpretability and robustness can call on the prototype-based models here-in utilized respectively for predictions or an ensemble thereof. Users whose work entails non-prototype-based models may opt for the deep CNN ensemble model. 


## Prototype-based 

The malaria prototypes were trained with the ```ImageGLVQ```, ```ImageGMLVQ```, ```ImageGMLVQ warmed with ImageGLVQ``` prototype-based models. The prototype initialization was done uniformly on one prototype per class basis. Predictions for each malaria test case is based on the trained protypes displayed below.


##### Initialised prototypes for  parasite class and uninfected class after preprocessing transformation

![download](https://user-images.githubusercontent.com/82911284/175720641-109baf6d-653f-435d-8498-bde91a36ab7a.png)

### ImageGLVQ model

Trained Prototypes of parasite class and uninfected class from ImageGLVQ model

![download](https://user-images.githubusercontent.com/82911284/175665273-fca57a7f-f701-4e6f-8708-0071c6141a9a.png)

### ImageGMLVQ model
Trained  Prototypes of parasites class and uninfected class from ImageGMLVQ model

![download](https://user-images.githubusercontent.com/82911284/175665381-fb6b1c5a-146b-4e6e-a647-a006e15dff00.png)


### ImageGMLVQ model warmed with ImageGLVQ model prototypes
Trained Prototypes of parasites class and uninfected class from the ImageGMLVQ model warmed with the ImageGLVQ learned prototypes

![download](https://user-images.githubusercontent.com/82911284/175665202-5df00dda-de61-43dc-8dc9-8162cfa07fcb.png)


## Non-Prototype-based

A deep convolutional neural network model built on transfer learning with ```InceptionResNetV2``` architecture with some custom adjustments was used to train the malaria cell image data for the parasite and uninfected class.




