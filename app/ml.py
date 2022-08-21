import json
import os
import pickle
import re

import numpy as np
import pymorphy2
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from app.schemas import *

ML_SOURCES = os.getenv("ML_SOURCES", "./assets")

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


class ML:
    thr = 0.05

    def __new__(cls, *args, **kwargs):
        cls = super().__new__(cls, *args, **kwargs)
        cls.mean_model = pickle.load(open(f"{ML_SOURCES}/mean_model.pkl", "rb"))
        cls.all_model = pickle.load(open(f"{ML_SOURCES}/all_model.pkl", "rb"))
        cls.target_model = pickle.load(open(f"{ML_SOURCES}/target_model.pkl", "rb"))

        cls.mean_vectorizer = pickle.load(
            open(f"{ML_SOURCES}/mean_vectorizer.pkl", "rb")
        )
        cls.knn_model_props = pickle.load(
            open(f"{ML_SOURCES}/knn_model_props.pkl", "rb")
        )

        cls.label2id = json.load(open(f"{ML_SOURCES}/label2id.json", "r"))

        return cls

    def predict(self, data: list[ProductPredict]) -> list[Product]:
        data = [i.dict() for i in data]
        for i, v in enumerate(data):
            data[i]["name"] = self._data_prepare([v["name"]], ru=True)
            data[i]["props"] = self._data_prepare([" ".join(v["props"])], en=True)

        return [self._predict(i["id"], i["name"], i["props"]) for i in data]

    @staticmethod
    def _data_prepare(language_dict: list[str], ru=False, en=False, es=False):
        if ru:
            stop_words = set(stopwords.words("russian"))
            lemmatizer = pymorphy2.MorphAnalyzer()
        elif en:
            stop_words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
        elif es:
            stop_words = set(stopwords.words("spanish"))
            lemmatizer = nltk.stem.SnowballStemmer("spanish")
        else:
            raise Exception("Lang no support")

        dict_prepared = []

        for text in tqdm(language_dict):
            text = text.lower()
            numbers = re.findall(r"[0-9]+", text)

            word_tokens = word_tokenize(text)
            word_tokens = [w for w in word_tokens if not w in stop_words]
            if ru:
                word_tokens = [lemmatizer.parse(w)[0].normal_form for w in word_tokens]
            elif en:
                word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]
            elif es:
                word_tokens = [lemmatizer.stem(w) for w in word_tokens]

            word_tokens = [w for w in word_tokens if not w in stop_words]

            filtered_text = " ".join(word_tokens + numbers)
            dict_prepared.append(filtered_text)
        return np.array(dict_prepared)

    def _predict(self, id_: str, name: str, props: str) -> Product:
        target_pred = self.target_model.predict_proba(name)
        mean_pred = self.mean_model.predict_proba(self.mean_vectorizer.transform(name))
        all_pred = self.all_model.predict_proba(name)

        pred = np.argmax(np.mean([target_pred, mean_pred, all_pred], axis=0), axis=1)

        dist, ind = self.mean_model.kneighbors(
            self.mean_vectorizer.transform(name), n_neighbors=5, return_distance=True
        )
        relative_dist = dist[:, 1] - dist[:, 0]
        dist = np.array(dist)
        ind = np.array(ind)

        dist_props, ind_props = self.knn_model_props[1].kneighbors(
            self.knn_model_props[0].transform(props),
            n_neighbors=471,
            return_distance=True,
        )

        mean_label = np.arange(0, 471)
        for i in range(len(pred)):
            if relative_dist[i] < 0.05:
                best_idx = ind_props[i][np.isin(ind_props[i], mean_label[ind[i]])][0]
                pred[i] = best_idx

        reference_ids = [[self.label2id[str(x)] for x in label] for label in ind]
        scores = 1 - dist

        return Product(
            id=id_,
            reference_id=reference_ids and reference_ids[0][0],
            scores=[
                ProductScore(reference_id=ref_id, score=distance)
                for ref_id, distance in zip(*reference_ids, *scores)
            ],
        )
