# 고용노동부 K디지털 아카데미 크롤링
import requests
from bs4 import BeautifulSoup
import pandas as pd


program_list = [] # 프로그램 목록
program_info = [] # 각 프로그램 정보
end_page = 54


# 프로그램 리스트 만들기
def make_list():
	for page in range(1, end_page+1):
		no = str(page)
		url = 'https://www.work24.go.kr/hr/a/a/1100/trnnCrsInf2.do?dghtSe=A&traingMthCd=A&tracseTme=2&endDate=20260221&keyword1=&keyword2=&pageSize=10&orderBy=ASC&startDate_datepicker=2025-02-21&currentTab=1&topMenuYn=&pop=&tracseId=AIG20240000459024&pageRow=10&totamtSuptYn=A&keywordTrngNm=&crseTracseSeNum=&keywordType=1&gb=&keyword=&kDgtlYn=&area=00%7C%EC%A0%84%EA%B5%AD+%EC%A0%84%EC%B2%B4&orderKey=2&mberSe=&kdgLinkYn=&srchType=all_type&crseTracseSe=A%7C%EB%94%94%EC%A7%80%ED%84%B8%EC%8B%A0%EA%B8%B0%EC%88%A0%EB%B6%84%EC%95%BC+%EC%A0%84%EC%B2%B4&tranRegister=&mberId=&i2=A&pageId=6&programMenuIdentification=EBG020000000313&crseTracseSeKDT=A_KDT%7C%ED%9B%88%EB%A0%A8%EC%9C%A0%ED%98%95+%EC%A0%84%EC%B2%B4&endDate_datepicker=2026-02-21&monthGubun=&pageOrder=2ASC&pageIndex='+ no +'&bgrlInstYn=&startDate=20250221&ncs=&gvrnInstt=&selectNCSKeyword=&action=trnnCrsInf2Post.do'

		html = requests.get(url)
		soup = BeautifulSoup(html.text,	'html.parser')

		company = soup.select('div.company_title.al_center a')
		program = soup.select('h3')
		
		# 링크 주소 추출
		for l in company:
			link = l.get('onclick')
			link = link.split(',')

		# 회사명, 프로그램명, 링크 리스트 추가 
		for c, p, l in zip(company, program, company):
			link = l.get('onclick').split("'")
			link = [link[i] for i in range(1, len(link), 2)]
			program_list.append([c.text.strip(), p.text.strip(), link])
			# program_word_cloud.append(p.text.strip())

make_list()
# print(program_list)



# 각 프로그램 정보 크롤링
def make_program_info():
	for i in range(len(program_list)):
		link1 = program_list[i][2][0]
		link2 = program_list[i][2][1]
		link3 = program_list[i][2][3]
		link4 = program_list[i][2][2]

		url = 'https://hrd.work24.go.kr/hrdp/co/pcobo/PCOBO0100P.do?tracseId='+ link1 + '&tracseTme=' + link2 +'&crseTracseSe=' + link3 + '&trainstCstmrId=' + link4 + '&tracseReqstsCd=undefined&cstmConsTme=undefined#undefined'
		# url_list.append(url) # url 리스트에 추가

		html = requests.get(url)
		soup = BeautifulSoup(html.text,	'html.parser')

		ncs = soup.select('div.infoList li span.con')
		# ncs_list.append(ncs[1].text.strip()) # NCS 분류 추가
		content = soup.select('tbody')
		program_info.append([ncs[1].text.strip(), content[0].text.strip().replace('\n',''), url])


make_program_info()
# print(program_info)


# 프로그램 정보 데이터프레임으로 변환 및 csv파일로 저장
def make_df():
	gov24_df1 = pd.DataFrame(data=program_list, columns=['회사명','프로그램명','주소'])
	gov24_df1.drop('주소', axis=1, inplace=True)
	gov24_df2 = pd.DataFrame(data=program_info, columns=['NCS분류','내용','주소'])
	gov24_df = pd.concat([gov24_df1, gov24_df2], axis = 1)

	gov24_df.to_csv('gov24.csv', encoding="utf-8-sig") # csv 파일로 저장
	return print(gov24_df)

make_df()