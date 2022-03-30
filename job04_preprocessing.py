import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from konlpy.tag import Okt

df = pd.read_csv('./crawling_data/naver_news_220330.csv')
# print(df.head())
# print(df.category.value_counts())
# df.info()
X = df.title
Y = df.category

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y) # encoder
# print(encoder.classes_)
# print(labeled_Y[:5])
