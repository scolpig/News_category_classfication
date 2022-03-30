
import pandas as pd

Politics= pd.read_csv('./naver_news_politics.csv', sep=',',index_col=0)

Economic= pd.read_csv('./naver_news_Economic.csv', sep=',',index_col=0)
Social= pd.read_csv('./naver_news_Social.csv', sep=',',index_col=0)
Culture= pd.read_csv('./naver_news_Culture_20220330.csv', sep=',',index_col=0)
IT= pd.read_csv('./naver_news_IT_20220330.csv', sep=',',index_col=0)
World= pd.read_csv('./naver_news_World_20220330.csv', sep=',',index_col=0)



df_sum = pd.DataFrame()

df_sum= pd.concat([Politics, Economic, Social, Culture, IT, World ], axis='rows', ignore_index=True)

print(df_sum.head())
print(df_sum.info())
df_sum.to_csv('./naver_news_220330.csv', index = False)

