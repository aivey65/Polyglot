import pandas as pd
import numpy as np
import regex as re
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

TOP_LANG = ["eng", "cmn", "hin", "spa", "fra", "ara", "ben", "rus", "por", "ind", "urd", "deu", "jpn", "swh", "pnb", "tam", "kor", "vie", "jav", "ita", "tha", "tgl", "pol", "yor", "ukr", "ibo", "npi", "ron", "nld", "zsm", "afr", "grc", "swe", "heb", "lat", "san", "gle", "mri", "chr", "nav", "haw", "smo"]

def preprocessText(text):
    pieceTokenizer = joblib.load("MLModel/word_piece.joblib")

    text = re.sub(r"(?=[\p{Common}])[^']|(?<![a-zA-Z])'|'(?![a-zA-Z])", " ", text.lower())
    line = text.strip()

    if " " not in line:
        line = " ".join(pieceTokenizer.encode(line).tokens)

    return line

def returnSelf(x):
    return x

def createModel(fileUrl="language_corpus.csv"):
    pieceTokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    pieceTokenizer.pre_tokenizer = Whitespace()
    pieceTrainer = WordPieceTrainer(vocab_size=50000, special_tokens=["[UNK]"])
    pieceTokenizer.train([fileUrl], pieceTrainer)

    data = pd.read_csv(fileUrl)

    train_text = data.apply(lambda x: " ".join(pieceTokenizer.encode(x['sentence']).tokens) if " " in x['sentence'] else x["sentence"], axis=1)

    le = LabelEncoder()
    le.fit(data["lan_code"])
    labels = le.transform(data["lan_code"])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(train_text)
    train_text = vectorizer.transform(train_text)

    languageDetectionModel = ComplementNB()
    languageDetectionModel.fit(train_text, labels)

    joblib.dump(languageDetectionModel, "MLModel/language_detection_model.joblib")
    joblib.dump(vectorizer, "MLModel/vectorizer.joblib")
    joblib.dump(le, "MLModel/label_encoder.joblib")
    joblib.dump(pieceTokenizer, "MLModel/word_piece.joblib")

def testModel(fileUrl="language_corpus.csv"):
    pieceTokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    pieceTokenizer.pre_tokenizer = Whitespace()
    pieceTrainer = WordPieceTrainer(vocab_size=50000, special_tokens=["[UNK]"])
    pieceTokenizer.train([fileUrl], pieceTrainer)

    data = pd.read_csv(fileUrl)
    data["sentence"] = data.apply(lambda x: " ".join(pieceTokenizer.encode(x['sentence']).tokens) if " " in x['sentence'] else x["sentence"], axis=1)

    le = LabelEncoder()
    le.fit(data["lan_code"])
    labels = le.transform(data["lan_code"])

    train_text, test_text, train_labels, test_labels = train_test_split(data["sentence"], labels, test_size=0.2)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(train_text)
    train_text = vectorizer.transform(train_text)
    test_text = vectorizer.transform(test_text)

    languageDetectionModel = ComplementNB()
    languageDetectionModel.fit(train_text, train_labels)

    predictions = languageDetectionModel.predict(test_text)

    return accuracy_score(test_labels, predictions)

def createPrediction(prediction_text, fileUrl="language_corpus.csv"):
    languageDetectionModel, vectorizer, labelEncoder = pullMLResources(fileUrl)

    prediction_text = vectorizer.transform([preprocessText(prediction_text)])
    result = languageDetectionModel.predict(prediction_text)

    return labelEncoder.inverse_transform(result)

def pullMLResources(corpusFile="language_corpus.csv"):
    languageDetectionModel = None
    vectorizer = None
    labelEncoder = None

    try:
        languageDetectionModel = joblib.load("MLModel/language_detection_model.joblib")
        vectorizer = joblib.load("MLModel/vectorizer.joblib")
        labelEncoder = joblib.load("MLModel/label_encoder.joblib")
    except OSError as e:
        createModel(corpusFile)
        languageDetectionModel = joblib.load("MLModel/language_detection_model.joblib")
        vectorizer = joblib.load("MLModel/vectorizer.joblib")
        labelEncoder = joblib.load("MLModel/label_encoder.joblib")

    return languageDetectionModel, vectorizer, labelEncoder