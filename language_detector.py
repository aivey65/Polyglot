import pandas as pd
import numpy as np
import regex as re
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

TOP_LANG = ["eng", "cmn", "hin", "spa", "fra", "ara", "ben", "rus", "por", "ind", "urd", "deu", "jpn", "swh", "pnb", "tam", "kor", "vie", "jav", "ita", "tha", "tgl", "pol", "yor", "ukr", "ibo", "npi", "ron", "nld", "zsm", "afr", "grc", "swe", "heb", "lat", "san", "gle", "mri", "chr", "nav", "haw", "smo"]

languageDetectionModel = None
cv = None

def preprocessData(fileUrl="language_data.csv"):
    data = pd.read_csv(fileUrl)

    data.drop(["id"], axis=1, inplace=True)
    data["sentence"] = data.apply(lambda x: re.sub(r"(?=[\p{Common}])[^']|(?<![a-zA-Z])'|'(?![a-zA-Z])", " ", x["sentence"].lower()), axis=1)

    filtered_data = data[data["lan_code"].isin(TOP_LANG)]

    """ To enforce all languages have an equal number of samples:"""
    # filtered_data = filtered_data.groupby("lan_code").sample(28)

    """ To just ensure that language sample sizes are not larger than 800:"""
    filtered_data = filtered_data[filtered_data.groupby("lan_code").cumcount() < 1000]

    data.to_csv(fileUrl, encoding='utf-8', index=False)

def createModel(fileUrl="language_data.csv"):
    data = pd.read_csv(fileUrl)
    train_text = data["sentence"]

    le = LabelEncoder()
    labels = le.fit_transform(data["lan_code"])

    cv = CountVectorizer(ngram_range=(1,4))
    cv.fit(train_text)
    train_text = cv.transform(train_text)

    languageDetectionModel = ComplementNB()
    languageDetectionModel.fit(train_text, labels)

    joblib.dump(languageDetectionModel, "MLModel/language_detection_model.joblib")
    joblib.dump(cv, "MLModel/vectorizer.joblib")

def splitData(text_data, labels, test_size=0.2):
    return train_test_split(text_data, labels, test_size=0.2)

def testModel(fileUrl="language_data.csv"):
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

def createPrediction(prediction_text, fileUrl="language_data.csv"):
    languageDetectionModel = None
    cv = None

    try:
        languageDetectionModel = joblib.load("MLModel/language_detection_model.joblib")
        cv = joblib.load("MLModel/vectorizer.joblib")
    except Exception as e:
        createModel(fileUrl)
        languageDetectionModel = joblib.load("MLModel/language_detection_model.joblib")
        cv = joblib.load("MLModel/vectorizer.joblib")

    prediction_text = cv.transform([prediction_text])

    return languageDetectionModel.predict(prediction_text)