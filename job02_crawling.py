from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time

def crawl_title():
    try:

        title = driver.find_element_by_xpath('//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[2]/a'.format(j, i)).text
        title = re.compile('[^가-힣a-zA-Z ]').sub(' ', title)
        title_list.append(title)
    except NoSuchElementException:
        print('NoSuchElementException')


option = webdriver.ChromeOptions()
# options.add_argument('headless')  이거 활성화하면 웹 브라우저가 안 뜸. 보고싶으면 주석 풀면 되고. 근데지금은 주석하래. 지금은 이거 하면 에러뜬대

option.add_argument('lang=ko_KR')
option.add_argument('--no-sandbox')   #이 아래 3개는 맥 어쩌고에서 필요한 거임. 윈도우 주석 풀어놔도 괜찮음.
option.add_argument('--disable-dev-shm-usage')
option.add_argument('disable-gpu')

driver= webdriver.Chrome('./chromedriver', options = option)
driver.implicitly_wait(10)


category = ['Politics', 'Economic', 'Social',
            'Culture',  'IT', 'World']

page_num = [140, 374, 486, 71, 76, 125]


#https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=1
# //*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[2]/dl/dt[2]/a    ul 1~4 / li 1~5


df_title = pd.DataFrame()

for l in range(0,1):
    title_list = []
    for k in range(1, page_num[l]+1):
    # for k in range(1, 3):
        url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page={}'.format(l, k)

        driver.get(url)
        time.sleep(0.5)
        for j in range(1, 5):
        # for j in range(1, 3):
            for i in range(1, 6):
            # for i in range(1, 3):
                try:
                    crawl_title()
                except StaleElementReferenceException:
                    # print('StaleElementReferenceException')
                    driver.get(url)
                    time.sleep(0.5)
                    crawl_title()
                except:
                    print('error')


    df_section_title = pd.DataFrame(title_list, columns= ['title'])
    df_section_title['category'] = category[l]
    df_title= pd.concat([df_title, df_section_title], axis='rows', ignore_index=True)

driver.close()
df_title.info()
print(df_title.category.value_counts())
df_title.to_csv('./crawling_data/naver_news_politics.csv')

