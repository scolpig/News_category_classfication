import pandas as pd
import os

forders = os.listdir('crawling_data')
print(forders)
category = ['Politics', 'Economic', 'Social',
            'Culture', 'World', 'IT']
df_all = pd.DataFrame()

print(forders[2].split('_')[2:3])
print(forders[3].split('_')[:2])
print(forders[3].split('_')[:3])
print(forders[4].split('_')[2:3])
print(forders[5].split('_')[2:3])

# for i in range(0,len(forders)):
#     if forders[i].split('_')[2] in category:
#         file = 'crawling_data/'+forders[i]
#         df= pd.read_csv(file,encoding='utf-8', index_col=0)
#         df_all = pd.concat([df_all, df], axis='rows', ignore_index=True)
# df_all.to_csv('./crawling_data/all.csv')


#애초에 저장할때 csv', index=False 이렇게 저장해서 오던가
#불러올때 index_col=0으로 0번째 컬럼을 인덱스로 쓰라는 말