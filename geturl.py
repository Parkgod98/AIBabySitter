import requests
import base64

def get_url(img_path) :

    # Imgur API 클라이언트 ID
    client_id = '-----------' # imgur에서 받은 Client Key -> 가림

    # 업로드할 이미지 파일
    image_path = img_path # 입력 받은 image path

    # 이미지를 base64 형식으로 인코딩
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read())

    # Imgur API에 이미지 업로드 요청
    response = requests.post(
        'https://api.imgur.com/3/image',
        headers={'Authorization': f'Client-ID {client_id}'},
        data={'image': image_data}
    )

    # 응답에서 이미지 URL 추출
    image_url = response.json()['data']['link']
    print(f'Uploaded image URL: {image_url}')
    return image_url