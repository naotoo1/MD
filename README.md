# MD-Flask
End-to-end implementation of Malaria detection using advanced highly-interpretable and robust prototype-based ML models (Image Generalized Learning Vector Quantization and  Image Generalized Matrix Learning Vector Quantization) as well as a deep learning model built on the transfer learning of Visual Geometry Group 19 convolutional neural network within the streamlit web app, dockerized and deployed on Azure infrastructure as a service cloud.

## What is it?
MD-Flask ML webapp is used for detecting malaria based on the parasites class and uninfected class. 

The implementation entails the use of prototype-based models(```ImageGLVQ```, ```ImageGMLVQ```, ```ImageGMLVQ warmed with ImageGLVQ```) and a deep learning model built by transfer learning with ```VGG19``` convolutional neural network deployed with respective use for the models, ensemble use for the prototype-based models and ensemble use of prototype-based cum CNN model built with transfer learning.

## Prototype-based 
Initialised prototypes for  parasite class and uninfected class after preprocessing transformation

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

A deep convolutional neural network model built by transfer learning with ```InceptionResNetV2``` architecture with some custom adjustments has been was employed in the trainng of the malaria data with a validation test accuracy of 96.26%

![image](https://user-images.githubusercontent.com/82911284/175787935-e8de6f04-e85f-461d-bf79-ffac4b62358d.png)



![image](https://user-images.githubusercontent.com/82911284/175787647-fc29e0b0-02d5-4f13-a638-a0d21a8d084d.png)






