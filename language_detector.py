import pandas as pd
import numpy as np
import regex as re
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

TOP_LANG = ["eng", "cmn", "hin", "spa", "fra", "ara", "ben", "rus", "por", "ind", "urd", "deu", "jpn", "swh", "pnb", "tam", "kor", "vie", "jav", "ita", "tha", "tgl", "pol", "yor", "ukr", "ibo", "npi", "ron", "nld", "zsm", "afr", "grc", "swe", "heb", "lat", "san", "gle", "mri", "chr", "nav", "haw", "smo"]

languageDetectionModel = None
cv = None

def preprocessText(text):
    text = re.sub(r"(?=[\p{Common}])[^']|(?<![a-zA-Z])'|'(?![a-zA-Z])", " ", text.lower())
    if " " not in text:
        " ".join(text)

    return text


def createModel(fileUrl="language_corpus.csv"):
    data = pd.read_csv(fileUrl)
    train_text = data["sentence"]

    le = LabelEncoder()
    le.fit(data["lan_code"])
    labels = le.transform(data["lan_code"])

    # tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    # tokenizer.pre_tokenizer = Whitespace()
    # trainer = WordPieceTrainer(vocab_size=1000000, special_tokens=["[UNK]"])
    # tokenizer.train(data["sentences"], trainer)
    # tokenizer.save("MLModel/tokenizer")

    cv = CountVectorizer(ngram_range=(1,4))
    cv.fit(train_text)
    train_text = cv.transform(train_text)

    languageDetectionModel = ComplementNB()
    languageDetectionModel.fit(train_text, labels)

    joblib.dump(languageDetectionModel, "MLModel/language_detection_model.joblib")
    joblib.dump(cv, "MLModel/vectorizer.joblib")
    joblib.dump(le, "MLModel/label_encoder.joblib")

def splitData(text_data, labels, test_size=0.2):
    return train_test_split(text_data, labels, test_size=0.2)

def testModel(fileUrl="language_corpus.csv"):
    data = pd.read_csv(fileUrl)
    text_data = data["sentence"]

    le = LabelEncoder()
    labels = le.fit_transform(data["lan_code"])

    train_text, test_text, train_labels, test_labels = train_test_split(text_data, labels, test_size=0.2)

    cv = CountVectorizer(ngram_range=(1,4))
    cv.fit(train_text)
    train_text = cv.transform(train_text)
    test_text = cv.transform(test_text)

    model = ComplementNB()
    model.fit(train_text, train_labels)

    predictions = model.predict(test_text)

    return accuracy_score(test_labels, predictions)

def createPrediction(prediction_text, fileUrl="language_corpus.csv"):
    languageDetectionModel = None
    cv = None
    labelEncoder = None

    try:
        languageDetectionModel = joblib.load("MLModel/language_detection_model.joblib")
        cv = joblib.load("MLModel/vectorizer.joblib")
        labelEncoder = joblib.load("MLModel/label_encoder.joblib")
    except Exception as e:
        createModel(fileUrl)
        languageDetectionModel = joblib.load("MLModel/language_detection_model.joblib")
        cv = joblib.load("MLModel/vectorizer.joblib")
        labelEncoder = joblib.load("MLModel/label_encoder.joblib")

    prediction_text = cv.transform([preprocessText(prediction_text)])
    result = languageDetectionModel.predict(prediction_text)

    return labelEncoder.inverse_transform(result)