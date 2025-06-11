# ============================== 
# 식단관리 프로그램
# ==============================

# 경북대 음식점 메뉴
knu = {'생연어덮밥':600, '들기름 우동':500, '오픈샌드위치':240, '봉골레 파스타': 500, '콰트로 포르마지 피자':889, '짬뽕':690, '크림리조또':446, '두루치기':436, '김치찌개':250}

while True:
    
    # 메뉴 출력
    print('='*90)
    print(f'{"  점심 식단관리 프로그램  ":=^80}'.center(80))
    print('='*90)
    print("1. 개인정보 입력하기".center(80))
    print("    2. 식단조절 필요성 확인하기".center(80))
    print("  3. 칼로리 계산하기  ".center(80))
    print(" 4. 저녁 메뉴 추천받기".center(80))
    print("5. 종료하기    ".center(80))
    print('='*90)
 
    # 메뉴 선택
    choice = input("메뉴 선택 >>> ")
    
    # 프로그램 종료하기
    if choice == '5':
        print(f'\n\n\n{"="*90}')
        print("프로그램을 종료합니다.".center(80))
        print(f'{"="*90}\n\n\n')
        break
    
    # 메뉴에 따른 기능 코드 실행
    # 1. 개인정보 입력하기
    if choice == '1':
        print(f'\n\n\n{"="*90}')
        personal_Info = input("성별, 키, 몸무게를 입력해주세요(입력 예시: 여자/160/60)\n>>> ").split('/')
        print(f'{"="*90}\n\n\n')
        
        gender = personal_Info[0]
        height = float(personal_Info[1])
        weight = float(personal_Info[2])    
  
        if gender == '남자':
            num = 22
        elif gender == '여자':
            num = 21
        standard_weight = height * 0.01 * height * 0.01 * num # 표준 체중
        need_kcal = standard_weight * 30 # 하루 섭취 가능 칼로리
        
    # 2. 식단조절 필요성 확인하기
    elif choice == '2':
        if weight >= standard_weight:
            diet_need = True
            print(f'\n\n\n{"="*90}')
            print(f"당신의 몸무게는 {weight:.1f}kg이고, 표준 체중은 {standard_weight:.1f}kg 입니다.\n{weight-standard_weight:.1f}kg만큼 감량이 필요합니다.")
            print(f'{"="*90}\n\n\n')
        else:
            diet_need = False
            print(f'\n\n\n{"="*90}')
            print(f"당신의 몸무게는 {weight:.1f}kg이고, 표준 체중은 {standard_weight:.1f}kg 입니다.\n당신은 식단조절을 할 필요가 없습니다:)")
            print(f'{"="*90}\n\n\n')

    # 3. 칼로리 계산하기
    elif choice == '3':
        # 감량 유무에 따른 하루 섭취 가능 칼로리 구하기
        if diet_need: # 체중감량이 필요한 사람의 경우
            energy = need_kcal - 300            
        else: # 체중감랑이 필요 없는 사람의 경우
            energy = need_kcal
        print(f'\n\n\n{"="*90}')
        print(f'당신의 하루 섭취 가능한 칼로리는 {energy:.1f}kcal입니다.')
        print(f'{"="*90}')
        5        
        
        # 점심 칼로리 입력받기    
        lunch_kcal = [] # 점심 칼로리 목록
        while True:
            user_lunch = input("칼로리를 입력해주세요.(입력 예시: 500)\n입력이 끝나면 <끝>이라고 적어주세요.\n>>> ")         
            if user_lunch == "끝":
                print(f'\n\n\n{"="*90}')
                print("점심 메뉴 입력을 종료합니다.")
                print(f'{"="*90}\n\n\n')
                break
            else:
                if user_lunch.isdecimal(): # 입력받은 칼로리가 숫자인지 확인
                    lunch_kcal.append(int(user_lunch)) 
                else:
                    print(f'\n\n\n{"="*90}')
                    print("칼로리를 숫자로 입력해주세요.")
                    print(f'{"="*90}\n\n\n')

        # 저녁에 섭취 가능한 칼로리
        left_kcal = energy - sum(lunch_kcal)
        print(f'\n\n\n{"="*90}')
        print(f'저녁에 섭취 가능한 칼로리는 {left_kcal:.1f}kcal입니다.')
        print(f'{"="*90}\n\n\n')
          
    # # 4. 저녁메뉴 추천받기
    elif choice == '4':
        if left_kcal > 0:
            dinner_kcal = list(knu.values()) # 맛집 칼로리 값만 추출해 칼로리 리스트 만들기
            recommend_list = [] 
            # 칼로리 리스트에서 저녁에 섭취 가능한 칼로리 보다 적은 칼로리만 추출
            for kcal in dinner_kcal:
                if kcal <= left_kcal:
                    recommend_list.append(kcal)
                    
            # 칼로리 값을 키값으로 바꾸기
            for idx in range(len(recommend_list)):
                recommend = {key:value for key, value in knu.items() if value in recommend_list}
            print(f'\n\n\n{"="*90}')
            print(f"오늘 저녁은 {list(recommend.keys())} 어떠신가요?")
            print(f'{"="*90}\n\n\n')
        else:
            print(f'\n\n\n{"="*90}')
            print("오늘 섭취할 칼로리를 모두 섭취하셨습니다.")
            print(f'{"="*90}\n\n\n')