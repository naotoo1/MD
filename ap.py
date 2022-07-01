# -*- coding: utf-8 -*-

import prototorch as pt
import torch.utils.data
import prototorch.models
import torchvision.transforms as transforms
from PIL import Image
import os

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Initialise the model from the checkpoints
trained_model = pt.models.glvq.ImageGMLVQ.load_from_checkpoint(
    "imagegmlvq112.ckpt",
    strict=False)

# create the model
model = pt.models.glvq.ImageGMLVQ(
    dict(input_dim=12288,
         latent_dim=12288,
         distribution=(2, 1),
         proto_lr=0.0001,
         bb_lr=0.0001),
    optimizer=torch.optim.Adam,
    prototypes_initializer=pt.initializers.LCI(trained_model.prototypes),
    labels_initializer=pt.initializers.LLI(trained_model.prototype_labels),
)

# Load your trained model
ckpt_path = "imagegmlvq112.pth"
model.load_state_dict(torch.load(ckpt_path), strict=False)


def model_predict(img_path, model):
    # Preprocessing the image
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(), ])
    img = Image.open(img_path)
    x = transform(img)
    x = x.unsqueeze(0)
    prediction = model.predict(x)
    with torch.no_grad():
        if int(prediction) == 1:
            pred = f'Uninfected with {(model(x))[0][1] * 100} % confidence'
        else:
            pred = f'Infected with {(model(x))[0][0] * 100} % confidence'
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run()
