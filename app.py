from flask import Flask, render_template, request, redirect
import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import Word
from textblob import TextBlob
from keras.preprocessing.text import Tokenizer
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
import json
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)


class Preprocessing:
  def lowercasing(self, text):
    return text.lower()

  def punctuationRemoval(self, text):
    text = re.sub('['+string.punctuation+']', '', text)
    text = re.sub(' +', ' ', text)
    return text

  def remove_url(self, text):
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', text)
    return text

  def correct_text(self, text):
    return TextBlob(text).correct()

  def text_preprocessing(self, text):
    text = self.lowercasing(text)
    text = self.remove_url(text)
    text = self.punctuationRemoval(text)
    text = str(self.correct_text(text))
    text = str(Word(text).lemmatize())

    return text


with open('static/models/word-index-new.json') as f:
    data = json.load(f)
    word_index = tokenizer_from_json(data)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    id_label_names = {0: 'negatif', 1: 'netral', 2: 'positif'}
    if request.method == 'POST':
        text = request.form['input_text']
        filter = request.form['filter']
        learning_rate = request.form['learning-rate']
        dropout = request.form['dropout']
        dimension = request.form['dimension']

        model = load_model('static/models/'+ dimension +'dim/model_cnn_filter'+ filter+'_kernel3_lr'+ learning_rate+'_dropout'+ dropout+'_epoch150.h5')

        preprocessing = Preprocessing()
        clean_text = preprocessing.text_preprocessing(text)

        # sentences = word_tokenize(clean_text)

        sequences = word_index.texts_to_sequences([clean_text])

        input_cnn = pad_sequences(sequences, maxlen=int(dimension), padding='post')

        preds = model.predict(input_cnn)[0]

        pred_classes = np.argsort(preds)[-4:][::-1]
        classes = [id_label_names[i] for i in pred_classes]
        props = preds[pred_classes]

        result = {}
        for c, p in zip(classes, props):
            result[c] = round(p*100, 2)

    return render_template("index.html", pred_classes=pred_classes, result=input_cnn, predict=preds, classes=list(result.keys()), props=list(result.values()), sentence=clean_text)


if __name__ == "__main__":
    app.run(debug=True)
