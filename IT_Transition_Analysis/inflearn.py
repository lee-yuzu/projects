# 인프런 크롤링
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

field = ['it-programming', 'game-dev-all', 'data-science', 'artificial-intelligence', 'it']
e_learning = {}

driver = webdriver.Chrome()

for f in field:
    e_learning[f] = []
    for page in range(1, 6):
        if page == 1:
            driver.get('https://www.inflearn.com/courses/' + f + '?levels=level-1,level-2&sort=POPULAR')
        else:
            driver.get('https://www.inflearn.com/courses/' + f + '?levels=level-1,level-2&sort=POPULAR' + str(page))
        
        driver.implicitly_wait(5)
        
        name_list = driver.find_elements(By.CLASS_NAME, "mantine-Text-root.css-10bh5qj.mantine-b3zn22")
        for name in name_list:
            e_learning[f].append(name.text)

driver.quit()
print(e_learning)


# 가장 긴 길이에 맞춰서 빈 값을 채우기 위해 max_length 계산
max_length = max(len(courses) for courses in e_learning.values())

# 각 필드의 데이터 길이가 다를 수 있으므로, 빈 문자열로 채움
for key in e_learning:
    while len(e_learning[key]) < max_length:
        e_learning[key].append('')

# 딕셔너리에서 DataFrame 생성하고 csv 파일로 저장
inflearn_df = pd.DataFrame(e_learning)
inflearn_df.to_csv('inflearn_courses.csv', index=False, encoding='utf-8-sig')



