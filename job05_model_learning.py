import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './crawling_data/news_data_max_29_wordsize_12831.npy',
    allow_pickle=True)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(12831, 300, input_length=29))  #차원을 줄여라 300으로
#차원은 커지는데, 단어수는 정해져있어서 그럼 서로의 거리도 멀어지고, 차지하는 비율도 적어짐
#데이터 전체에 12831개의 단어에 대한 값을 생성,

#conv1d
#순서가 있는 데이터도 이미지처럼 인식이 가능? 빠르게 이미지처럼 위치관계를 보고 빠르게 판단
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1)) #문장길이를 줄이지 않음, 사실 없어도 됨 pool_size=1 이라는 의미없는 레이어 추기
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax')) #카테고리 6개니깐 마지막 dense는 6개 , 다중분류라서 softmax
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=100, epochs=10, validation_data=(X_test, Y_test))
model.save('./output/news_category_classification_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))  #리스트속 마지막 val_accuracy 만 저장

plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()