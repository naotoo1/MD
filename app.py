# import libraries
import os
import pickle
import cv2
import numpy as np
import torch.utils.data
from flask import Flask, request, render_template
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from prosemble import Hybrid
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# load the trained models
pickle_in1 = open("svm4.pkl", "rb")
pickle_in2 = open("rf.pkl", "rb")
pickle_in3 = open("rslvq.pkl", "rb")
pickle_in4 = open("xgb.pkl", "rb")

model1 = pickle.load(pickle_in1)
model2 = pickle.load(pickle_in2)
model3 = pickle.load(pickle_in3)
model4 = pickle.load(pickle_in4)
model5 = torch.load('celvq.pt')

# input size
SIZE = 224
# load the prior model feature map extractor
IR_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in IR_model.layers:
    layer.trainable = False


# function to return the posterior for predictions
def get_posterior(x, y_, z_):
    """
    :param x: Input data
    :param y_: prediction
    :param z_: model
    :return: prediction probabilities
    """
    z1 = z_.predict_proba(x)
    certainties = [np.max(i) for i in z1]
    cert = np.array(certainties).flatten()
    cert = cert.reshape(len(cert), 1)
    y_ = y_.reshape(len(y_), 1)
    labels_with_certainty = np.concatenate((y_, cert), axis=1)
    return np.round(labels_with_certainty, 4)


# prototype labels
proto_classes = np.array([0, 1])

# Instantiate an object for ensemble from the Hybrid class using prosemble package
ensemble = Hybrid(model_prototypes=None, proto_classes=proto_classes, mm=2, omega_matrix=None, matrix='n')

# Instantiate an object for celvq from the Hybrid class inorder to get the recall procedure
model5_ = Hybrid(model_prototypes=model5.prototypes.detach().numpy(), proto_classes=proto_classes, mm=2,
                 omega_matrix=None,
                 matrix='n')

# Instantiate an object for rslvq from the Hybrid class inorder to get the recall procedure
model3_ = Hybrid(model_prototypes=model3.w_, proto_classes=proto_classes, mm=2, omega_matrix=None,
                 matrix='n')


#  Get the results for test instances
def get_summary_results(x, y, z):
    """

    :param x: list of all predictions from the models
    :param y: list of all securities from the models
    :param z: list of of all the models
    :return:
    classification results along with the corresponding confidence
    """
    results = ''
    summary = []
    num_ = len(y)
    for (i, v) in enumerate(x):
        if i < 2 and v > 0.5:
            results = f"  {z[i]}, uninfected, {np.round(y[i][0] * 100, 2)}% "
        if i < 2 and v < 0.5:
            results = f"  {z[i]}, parasites, {np.round(y[i][0] * 100, 2)}% "
        if i >= 2 and i != num_ and v > 0.5:
            results = f"  {z[i]}, uninfected, {np.round(y[i][0][1] * 100, 2)}% "
        if i >= 2 and i != num_ and v < 0.5:
            results = f"  {z[i]}, parasites, {np.round(y[i][0][1] * 100, 2)}% "
        if i == num_ and v > 0.5:
            results = f"  {z[i]}, uninfected, {np.round(y[i][0][1] * 100, 2)}% "
        if i == num_ and v < 0.5:
            results = f"  {z[i]}, uninfected, {np.round(y[i][0][1] * 100, 2)}% "
        summary.append(results)
    summary = np.array(summary).reshape(len(summary), 1)
    return str(summary)


#  get the predictions along with the confidence
def get_results(x, y):
    if x[0] > 0.5:
        results = f"Uninfected with {np.round(y[0][1] * 100, 2)}% confidence"
    else:
        results = f"Parasites with {np.round(y[0][1] * 100, 2)}% confidence"
    return results


# get the predictions and  confidence of the soft  and hard ensemble models
def get_results_ens(x, y):
    if x[0] > 0.5:
        results = f"Uninfected with {np.round(y[0] * 100, 2)}% confidence"
    else:
        results = f"Parasites with {np.round(y[0] * 100, 2)}% confidence"
    return results


#  get the predictions and confidence for the celvq model
def get_results_(x, y):
    if x > 0.5:
        results = f"Uninfected with {np.round(y[0][1] * 100, 2)}% confidence"
    else:
        results = f"Parasites with {np.round(y[0][1] * 100, 2)}% confidence"
    return results


def model_predict(img_path, method):
    # Preprocessing the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img / 255.0
    input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
    input_img_feature = IR_model.predict(input_img)
    input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)

    # Get prediction from the prototype models
    pred1 = model1.predict(input_img_features)
    pred2 = model2.predict(input_img_features)
    pred3 = model3.predict(input_img_features)
    pred4 = model4.predict(input_img_features)
    pred5 = model5.predict(torch.Tensor(input_img_features)).detach().numpy().astype('int32')

    # Get of confidence of predicted results with optimised security_coef chosen as 2
    sec1 = get_posterior(x=input_img_features, y_=pred1, z_=model1)
    sec2 = get_posterior(x=input_img_features, y_=pred2, z_=model2)
    sec4 = get_posterior(x=input_img_features, y_=pred2, z_=model4)
    sec3 = model3_.get_security(x=np.array(input_img_features), y=2)
    sec5 = model5_.get_security(x=np.array(input_img_features), y=2)

    all_pred = np.array([pred1, pred3, [pred5]])
    all_sec = np.array([sec1, sec3, sec5])

    # prediction from the ensemble using hard voting
    prediction1 = ensemble.pred_prob(input_img_features, all_pred)
    # prediction from the ensemble using soft voting
    prediction2 = ensemble.pred_sprob(input_img_features, all_sec)
    # # confidence of the prediction using hard voting
    hard_prob = ensemble.prob(input_img_features, all_pred)
    # # confidence of the prediction using soft voting
    soft_prob = ensemble.sprob(input_img_features, all_sec)

    summary_pred = [prediction2, prediction1, pred1, pred2, pred4, pred5, pred3]
    summary_sec = [soft_prob, hard_prob, sec1, sec2, sec4, sec5, sec3]
    models = ['soft_ensemble', "hard_ensemble", "svm", 'rf', 'xgb', 'celvq', 'rslvq']

    if method == "soft_ensemble_model":
        return get_results_ens(x=prediction2, y=soft_prob)

    if method == "hard_ensemble_model":
        return get_results_ens(x=prediction1, y=hard_prob)

    if method == "svm":
        return get_results(x=pred1, y=sec1)

    if method == "random_forest_model":
        return get_results(x=pred2, y=sec2)

    if method == "xgb_model":
        return get_results(x=pred4, y=sec4)

    if method == "celvq_model":
        return get_results_(x=pred5, y=sec5)

    if method == "rslvq_model":
        return get_results(x=pred3, y=sec3)

    if method == "summary_of_models":
        return get_summary_results(x=summary_pred, y=summary_sec, z=models)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        models = request.form.get('models')
        print(models)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, models)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
