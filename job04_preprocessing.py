import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #pip install scikit-learn
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
labeled_Y = encoder.fit_transform(Y)
# print(encoder.classes_)
# print(labeled_Y[:5])
with open('./output/encoder.pickle','wb') as f:
    pickle.dump(encoder, f) #encoder를 저장


onehot_Y = to_categorical(labeled_Y)
# print(onehot_Y)

okt = Okt() #형태소 단위로 잘라줌
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True) #단어의 기본형태로 만들어줌, 접었다=접다
stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
print(X[1])
print('='*100)
# print(okt_morpy_X)

token = Tokenizer()
token.fit_on_texts(X) #토큰을 데이터에 맞춘다 라벨 정보를 딕셔너리 형태로 갖음

tokened_X = token.texts_to_sequences(X)
print(tokened_X[1])
with open('./output/news_token.pickle','wb') as f:
    pickle.dump(token, f)

wordsize = len(token.word_index) + 1  #토큰은 단어한테 1부터 숫자를 부여, 0은 우리가 쓸거라서 단어의 갯수에 1 더해줌
# print(token.word_index)

max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)
#앞에 채우는 이유는 0은 무의미한단어 앞보다 뒤에 가중치가 붙기 때문에
X_pad = pad_sequences(tokened_X, max)  #max값에 맞춰서 앞에 0을 채워줌
print(X_pad[:10])

X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad, onehot_Y, test_size=0.1)
xy = X_train, X_test, Y_train, Y_test
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
np.save('./crawling_data/news_data_max_{}_wordsize_{}'.format(max, wordsize),xy)