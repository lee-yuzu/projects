# 인프런 분석
from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import	Counter
import matplotlib.pyplot as	plt
import platform
import numpy as	np
from PIL import Image
import koreanize_matplotlib
import pandas as pd


inflearn_df = pd.read_csv('./inflearn_courses.csv')

# 불용어
stopwords = ["편", "위", "김영한", "합격", "를", "로", "얄코", "수", "입", "개", "강의", "중", "법",
			 "부트", "나", "피", "마", "홍", "정모", "리뉴얼", "및", "만큼", "페이지", "고법", "모든",
			 "과목", "빅", "테크", "포기", "자의", "개정안", "제풀이", "줄", "도", "배리", "블", "방",
			 "초", "오늘", "서류", "률", "성법", "토비", "티스", "버전", "이불", "타", "과", "추가", "군데",
			 "코", "절", "대강", "좌", "치킨", "콤보", "살", "불사", "부터", "주", "롤링", "진짜", "완전",
			 "가지", "남", "박사", "롱런", "슈퍼", "고도", "쥬신", "아카데미", "쌤", "스", "방",
			 "인슈", "서브", "스탠", "것", "블루", "블", "이크", "스파르타", "역", "리깅", "누구",
			 "클론", "선비", "비", "전공자", "무작정", "자료", "지망", "생", "무엇", "제로", "세상", "두고두고",
			 "기", "트", "꼬마", "마녀", "공짜", "과 외", "무료", "캐", "주얼", "번외편", "위해", "대한",
			 "알", "아두", "쓸모", "기능", "통한", "박치기", "어드벤쳐", "위니", "브", "월드", "수복", "권",
			 "바로", "생", "개정판", "치", "배", "더", "모두", "블로그", "제공", "완주", "의", "왼손", "본기",
			 "단", "가장", "자기", "소개", "별", "맞춤", "은", "이제", "제대로", "판", "주요", "중요", "바",
			 "성기", "내", "제일", "버", "당신", "달리", "맥주", "값", "보기", "너", "김", "포기", "최고", "강사", "첫",
			 "걸음", "거대", "부", "저자", "급", "최소한", "반영", "상편", "한국", "대신", "꼭", "선택",
			 "카카오", "임", "직원", "음", "만난", "포함", "지금", "당장", "반", "아무", "곰책", "론", "반드시",
			 "문서", "현재", "장", "입문", "활용"]

def color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl({:d},{:d}%, {:d}%)".format(np.random.randint(64,191),np.random.randint(42,46),np.random.randint(9,32)))

# hsl(191, 46%, 9%)
# hsl(64, 42%, 32%)

for col in inflearn_df.columns:
	# 내용 부분만 추출하기
	content = inflearn_df[col]
	content.to_csv(col+'.txt')
	
	# 워드 클라우드
	text = open('./'+col+'.txt', encoding='utf-8').read()
	okt = Okt()

	sentences_tag =	[]
	sentences_tag =	okt.pos(text)

	noun_list =	[]
	# tag가 명사인 단어들만 noun_adj_list에 추가
	for	word, tag in sentences_tag:
		if tag in ['Noun']:
			noun_list.append(word)

	counts = Counter(noun_list)
	tags = counts.most_common(500)
	print(tag)

	tag_dict = dict(tags)

	#불용어 처리
	for	stopword in	stopwords:
		if stopword in tag_dict:
			tag_dict.pop(stopword)

	print(tag_dict)

	path = r'c:\\Windows\Fonts\malgun.ttf'
	wc = WordCloud(font_path=path,	width=1000, height=1000,
					background_color="white", max_font_size=200,
					repeat=True, color_func = color_func)

	cloud = wc.generate_from_frequencies(dict(tag_dict))
	# 생성된 WordCloud를 test.jpg로 보낸다.
	#cloud.to_file("test.png")
	plt.figure(figsize=(10,	8))
	plt.title(col, pad=10, weight='bold')
	plt.axis('off')
	plt.imshow(cloud)
	plt.show()



