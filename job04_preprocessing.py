import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #scikit-learn
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from konlpy.tag import Kkma, Okt
pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('./crawling_data/naver_news_220330.csv')
# print(df.head())
# print(df.category.value_counts())
df.info()
X = df.title
Y = df.category

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
print(encoder.classes_)
# print(labeled_Y[:5])
# with open('./output/encoder.pickle', 'wb') as f:
#     pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)

okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)

stopwords =

print(X[1])










