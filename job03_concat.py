import pandas as pd
import glob

data_paths = glob.glob('./data/*')
print(data_paths)


df = pd.DataFrame()
for path in data_paths:
    df_temp = pd.read_csv(path, index_col=0)
    print(df_temp.columns)
    df = pd.concat([df, df_temp], axis='rows')

df.to_csv('./data/naver_news_all.csv', index=False)