import json
import requests
import geturl as g

def Send(path,emotion) :
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type" : "authorization_code",
        "client_id" : "-----------------------", # 내 rest api키. -> 가림
        "redirect_url" : "https://www.daum.net/", # 내가 설정한 리다이렉트 url
        "code" : "y2k2alMD7QpKvS9dxOJ_2R4cmrn0F235a2nljnSLKSrNMXlUalSAvAAAAAQKKiUQAAABj_fqHs5HueF-5ScOZw"  # 리다이렉트 뒤에서 얻어낸 정보
    }
    response = requests.post(url, data=data)
    tokens = response.json()

    # kakao_code.json 파일 저장
    with open("kakao_code.json", "w") as fp:
        json.dump(tokens, fp)

    print(tokens) # 토큰은 시간 지나면 만료되는데 또 redirect_url로 얻어내지말고, refresh token을 이용해서 얻어내기.

    headers = {
        "Authorization": "Bearer " + "b50bLEsxZr-U2V1DnvIgN5GTClMm1rdXAAAAAQo9dZsAAAGP9-tzzKhuWkW__Nqy" # access token
    }

    # 친구 목록 가져오기
    url = "https://kapi.kakao.com/v1/api/talk/friends" 
    result = json.loads(requests.get(url, headers=headers).text)
    friends_list = result.get("elements")

    # 잘 가져왔는지 확인
    print(friends_list)

    friend_id = friends_list[0].get("uuid")

    url= "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"

    # image url 획득
    urll = g.get_url(path)

    # 기본 template 정의
    template_object={
            "object_type": "feed",
            "content": {
                "title": f"{emotion}",
                "description": "아이를 확인해주세요.",
                "image_url": urll,
                "image_width": 640,
                "image_height": 640,
                "link": {
                    "web_url": "http://www.daum.net",
                    "mobile_web_url": "http://m.daum.net",
                    "android_execution_params": "contentId=100",
                    "ios_execution_params": "contentId=100"
                }
            },
        }

    good = {"web_url" : urll, "mobile_web_url": urll}

    data = {'receiver_uuids':'["{}"]'.format(friend_id),"template_object" : json.dumps(template_object)}

    response = requests.post(url, headers=headers, data=data)
    if response.json().get('result_code') == 0:
        print('메시지를 성공적으로 보냈습니다.')
    else:
        print('메시지 전송 실패 ㅠㅠ')
        