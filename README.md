# MD ML
End-to-end implementation of Malaria detection using a prior model feature map extractor based on the transfer learning architecture of ```InceptionResNetV2``` plus ```svm```, ```rfc```, ```xgbost```, ```rslvq``` and ```celvq``` as stand-alone models with options for ```soft ensemble``` and ```hard ensemble``` based on the [prosemble ML package](https://github.com/naotoo1/prosemble) using svm, rslvq and celvq robust prototype-based ML models.

## What is it?
MD ML Webapp is a tool for detecting malaria based on the parasites class and uninfected class. A case study of malaria image cell classification using a combination of pretrained deep-cnn model + traditional ML models with some prototype-based options.

## How to use
python 3.9 or later with all [requirement](https://github.com/naotoo1/MD/blob/main/requirements.txt) dependencies installed including  ```xgboost ``` and  ```opencv ```.

```python
git clone https://github.com/naotoo1/MD.git
cd MD
pip install -r requirements.txt
```
Download the ```celvq``` and ```svm``` trained models into the ```MD``` folder using the link https://we.tl/t-7UcQgLFuwr

## Run 
```python
python app.py
```


