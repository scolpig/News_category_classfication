from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time

def crawling_title():
    try:
        title = driver.find_element_by_xpath('//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[2]/a'.format(j, i)).text
        title = re.compile('[^가-힣a-zA-Z ]').sub(' ', title)
        title_list.append(title)
    except NoSuchElementException:
        title = driver.find_element_by_xpath('//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt/a'.format(j, i)).text
        title = re.compile('[^가-힣a-zA-Z ]').sub(' ', title)
        title_list.append(title)
        print('NoSuchElementException')

option = webdriver.ChromeOptions()
#options.add_argument('headless')
option.add_argument('lang=ko_KR')
option.add_argument('--no-sandbox')
option.add_argument('--disable-dev-shm-usage')
option.add_argument('disable-gpu')
driver = webdriver.Chrome('./chromedriver', options=option)
driver.implicitly_wait(10)

category = ['Politics', 'Economic', 'Social',
            'Culture', 'World', 'IT']
page_num = [140, 140, 140, 71, 76, 125]


#https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100
#https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=1
#//*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a
#//*[@id="section_body"]/ul[1]/li[2]/dl/dt[2]/a
#//*[@id="section_body"]/ul[2]/li[1]/dl/dt[2]/a
# //*[@id="section_body"]/ul[4]/li[5]/dl/dt[2]/a
df_title = pd.DataFrame()
for l in range(2, 3):
    title_list = []
    for k in range(1, page_num[l]+1):
        url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page={}'.format(l, k)
        driver.get(url)
        time.sleep(0.5)
        for j in range(1, 5):
            for i in range(1, 6):
                try:
                    crawling_title()
                except StaleElementReferenceException:
                    driver.get(url)
                    time.sleep(0.5)
                    crawling_title()
                except:
                    print('error')
    df_section_title = pd.DataFrame(title_list, columns=['title'])
    df_section_title['category'] = category[l]
    df_title = pd.concat([df_title, df_section_title],
                         axis='rows', ignore_index=True)
driver.close()
df_title.info()
print(df_title.category.value_counts())
# df_title.to_csv('./crawling_data/1.naver_news_title_20220330.csv')
df_title.to_csv('./crawling_data/naver_news_Social.csv', index=False)