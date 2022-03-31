import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #pip install scikit-learn
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import *
from konlpy.tag import Okt, Kkma

pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('./crawling_data/naver_headline_news220331.csv')
print(df.head())

X = df.title
Y = df.category

with open('./output/encoder.pickle','rb') as f:
    encoder = pickle.load(f)
labeled_Y = encoder.transform(Y)
#정해져있는 라벨 그데로 라벨링 작업할거면 transform
#라벨을 주고 그 라벨이 뭐냐 물어볼때는 리버스트랜스폼

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

with open('./output/news_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
#모델에서 max는 29임 추가로 수집한 데이터가 29가 넘으면 그냥 짤라야함

for i in range(len(tokened_X)):
    if len(tokened_X[i])>29:
        tokened_X[i] = tokened_X[i][:29]
X_pad = pad_sequences(tokened_X, 29)  #짧은애는 0으로 채워서(padding)
label = encoder.classes_
model = load_model('./output/news_category_classification_model_0.77546626329422.h5')
preds = model.predict(X_pad)
predicts = []
for pred in preds:
    predicts.append(label[np.argmax(pred)])

df['predict'] = predicts
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] == df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'
print(df['OX'].value_counts()/len(df))  #accuracy
for i in range(len(df)):
    if df['category'][i] != df['predict'][i]:
        print(df.iloc[i])
