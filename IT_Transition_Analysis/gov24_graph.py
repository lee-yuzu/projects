# 고용노동부 K디지털 아카데미 분석
from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import	Counter
import matplotlib.pyplot as	plt
import platform
import numpy as	np
from PIL import Image
import koreanize_matplotlib
import pandas as pd

# 불용어
stopwords = ['수료', '목 표', '기준', '진행', '여부', '졸', '국가', '사람', '위해',
			 '발급', '유', '요건', '통', '하나', '가지', '개월', '기간', '과목',
			 '통한', '강의', '자체', '개', '선호', '일', '부여', '별', '것', '중인',
			 '누구', '학', '각', '로', '차', '급', '타', '기', '프로', '주', '인', '나',
			 '로서', '제공', '계열', '이', '가점', '후', '점', '를', '시', '위', '포함',
			 '카드', '사항', '예정자', '자', '등', '생', '수', '이상', '관련', '및',
			 '분야', '통해', '기타',' 경우', '대상', '분', '중', '우선', '우대', '대한',
			 '전공자', '선발', '학습', '과정', '선수', '졸업', '직무', '전공', '풀', '훈련', '초',
			 '버스', '유니티', '양성', '취업', '덕트', '기반', '역량', '해당', '교육', '경우',
			 '사전', '마', '개발', '피', '반', '이어도', '가의', '노', '바로', '스', '브', '법',
			 '소자', '도', '끝내기', '한번', '학력', '자격', '보유', '평가', '기본', '경험', '활용',
			 '정보처리', '고등학교', '아래', '초대', '부트캠프', '프로그래밍', '지식', '개발자']

gov24_df = pd.read_csv('./gov24_Preprocessing.csv')

def color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl({:d},{:d}%, {:d}%)".format(np.random.randint(228,250),np.random.randint(15,38),np.random.randint(19,46)))

# 프로그램명 hsl(228, 15%, 46%) hsl(250, 38%, 19%)
# 내용 hsl(250, 38%, 19%) hsl(359, 47%, 53%)

def make_wordcloud(option):
	# 내용 부분만 추출하기
	content = gov24_df[option]
	content.to_csv('content.txt')

	# 워드 클라우드
	text = open('./content.txt', encoding='utf-8').read()
	okt = Okt()

	sentences_tag =	[]
	sentences_tag =	okt.pos(text)

	noun_list =	[]
	# tag가 명사인 단어들만 noun_adj_list에 추가
	for	word, tag in sentences_tag:
		if tag in ['Noun']:
			noun_list.append(word)

	counts = Counter(noun_list)
	tags = counts.most_common(200)
	print(tag)


	tag_dict = dict(tags)

	# 불용어 처리
	for	stopword in	stopwords:
		if stopword in tag_dict:
			tag_dict.pop(stopword)

	print(tag_dict)

	path = r'c:\\Windows\Fonts\malgun.ttf'
	#img_mask = np.array(Image.open('cloud.png'))
	wc = WordCloud(font_path=path,	width=800, height=800,
					background_color="white", max_font_size=200,
					repeat=True, colormap='inferno', color_func = color_func)
	
	#mask=img_mask,

	cloud = wc.generate_from_frequencies(dict(tag_dict))
	# 생성된 WordCloud를 test.jpg로 보낸다.
	# cloud.to_file('test.jpg')
	plt.figure(figsize=(10,	8))
	plt.axis('off')
	plt.imshow(cloud)
	plt.show()


# NCS 코드 막대그래프 만들기
def make_graph():

	# 개수가 10개 이상인 것만 추출해서 시리즈로 변환
	classification = gov24_df['NCS분류'].value_counts(ascending=True).loc[lambda x:x >10]

	# NCS 코드 제거
	y = []
	for i in classification.index:
		y.append(i.split('(')[0])

	# 가로 막대 그래프
	plt.figure(figsize=(10, 6))
	plt.barh(y, classification, color='#6fb848')

	# fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10,7))

	# ax1.barh(y, classification, color='#854442')
	# ax1.set_xlim(0, 30) 
	# ax1.grid(alpha=0.5, linestyle='--')

	# ax2.barh(y, classification, color='#854442')
	# ax2.set_xlim(150, 250)
	# ax2.grid(alpha=0.5, linestyle='--')

	# for idx, value in enumerate(classification):
	#     if value <= 30: 
	#         ax1.text(value, idx, str(value), size=10, va='center')
	#     elif value >= 250:
	#         ax2.text(value, idx, str(value), size=10, va='center')

	plt.title('NCS분류', pad=10, fontsize=15, weight='bold')
	plt.grid(alpha=0.5, linestyle='--')
	plt.tight_layout()
	plt.show()


def main():
	while True:
		print('===========================')
		print('워드클라우드와 그래프 만들기')
		print('===========================')
		option = input('컬럼명을 입력하세요(프로그램명, 내용, 종료): ')
		if option == '종료':
			print('프로그램 종료')
			break
		else:
			make_wordcloud(option)
			make_graph()


main()