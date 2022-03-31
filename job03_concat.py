import pandas as pd
import glob
data_paths = glob.glob('./crawling_data/*')
# print(data_paths)
df1 = pd.DataFrame()
for path in data_paths[4:9]:
    df_temp = pd.read_csv(path, index_col=0)
    df1 = pd.concat([df1, df_temp], ignore_index=True, axis='rows')
for path in data_paths[10:11]:
    df_temp = pd.read_csv(path, index_col=0)
    pd.concat([df1, df_temp], ignore_index=True, axis='rows')
# pd.concat([df1, df2], ignore_index=True, axis='rows')
df1.info()

df1.to_csv('./naver_news_sum.csv', index = False)