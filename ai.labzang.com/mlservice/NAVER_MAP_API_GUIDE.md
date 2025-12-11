# 네이버 지도 API 사용 가이드

## 📋 개요

mlservice에서 네이버 지도 API를 사용하여 주소와 좌표 간 변환 기능을 제공합니다.

## 🔑 API 키 설정

네이버 지도 API를 사용하려면 네이버 클라우드 플랫폼에서 API 키를 발급받아야 합니다.

### 1. 네이버 클라우드 플랫폼에서 API 키 발급

1. [네이버 클라우드 플랫폼](https://www.ncloud.com/) 접속
2. **AI·NAVER API** → **Application** 등록
3. **Geocoding** 및 **Reverse Geocoding** API 권한 활성화
4. **Client ID**와 **Client Secret** 발급

### 2. 환경 변수 설정

`docker-compose.yaml` 또는 `.env` 파일에 다음 환경 변수를 설정합니다:

```yaml
NAVER_CLIENT_ID=your_client_id
NAVER_CLIENT_SECRET=your_client_secret
```

**주의**: 네이버 지도 API는 네이버 Local Search API와 동일한 Client ID/Secret을 사용하지만, API 권한이 다릅니다.
- **Local Search API**: 검색 API 권한 필요
- **Geocoding API**: Geocoding API 권한 필요
- **Reverse Geocoding API**: Reverse Geocoding API 권한 필요

## 🚀 API 엔드포인트

### 1. Geocoding (주소 → 좌표)

주소를 입력하면 위도, 경도 좌표를 반환합니다.

**엔드포인트**: `GET /api/ml/seoul_crime/geocode`

**Query Parameters:**
- `query` (필수): 검색할 주소

**예시:**
```
GET http://localhost:9010/api/ml/seoul_crime/geocode?query=서울특별시 강남구 테헤란로 152
```

**응답 예시:**
```json
{
  "success": true,
  "message": "Geocoding 성공",
  "data": {
    "address": "서울특별시 강남구 테헤란로 152",
    "roadAddress": "서울특별시 강남구 테헤란로 152",
    "jibunAddress": "서울특별시 강남구 역삼동 737",
    "latitude": 37.5002,
    "longitude": 127.0364
  }
}
```

### 2. Reverse Geocoding (좌표 → 주소)

위도, 경도를 입력하면 주소를 반환합니다.

**엔드포인트**: `GET /api/ml/seoul_crime/reverse-geocode`

**Query Parameters:**
- `latitude` (필수): 위도
- `longitude` (필수): 경도

**예시:**
```
GET http://localhost:9010/api/ml/seoul_crime/reverse-geocode?latitude=37.5002&longitude=127.0364
```

**응답 예시:**
```json
{
  "success": true,
  "message": "Reverse Geocoding 성공",
  "data": {
    "address": "서울특별시 강남구 역삼동 테헤란로 152",
    "roadAddress": "서울특별시 강남구 테헤란로 152",
    "jibunAddress": "서울특별시 강남구 역삼동 737",
    "sido": "서울특별시",
    "sigungu": "강남구",
    "dong": "역삼동",
    "latitude": 37.5002,
    "longitude": 127.0364
  }
}
```

## 📝 Postman 테스트 방법

### Step 1: Geocoding 테스트

1. Postman에서 새 Request 생성
2. Method: **GET**
3. URL: `http://localhost:9010/api/ml/seoul_crime/geocode`
4. **Params** 탭에서:
   - `query`: `서울특별시 강남구 테헤란로 152`
5. **Send** 클릭

### Step 2: Reverse Geocoding 테스트

1. Postman에서 새 Request 생성
2. Method: **GET**
3. URL: `http://localhost:9010/api/ml/seoul_crime/reverse-geocode`
4. **Params** 탭에서:
   - `latitude`: `37.5002`
   - `longitude`: `127.0364`
5. **Send** 클릭

## 🔧 코드 사용 예시

### Python에서 사용

```python
from app.seoul_crime.seoul_naver_client import SeoulNaverClient

# 클라이언트 초기화
naver_client = SeoulNaverClient()

# Geocoding: 주소 → 좌표
result = naver_client.geocode("서울특별시 강남구 테헤란로 152")
if result:
    print(f"위도: {result['latitude']}, 경도: {result['longitude']}")

# Reverse Geocoding: 좌표 → 주소
result = naver_client.reverse_geocode(37.5002, 127.0364)
if result:
    print(f"주소: {result['address']}")
```

## ⚠️ 주의사항

1. **Rate Limit**: 네이버 지도 API는 초당 1회 호출 제한이 있습니다. 코드에 자동 대기 기능이 포함되어 있습니다.

2. **API 권한**: 네이버 클라우드 플랫폼에서 Geocoding 및 Reverse Geocoding API 권한을 활성화해야 합니다.

3. **에러 처리**: 
   - 401 에러: API 키 인증 실패 또는 권한 없음
   - 404 에러: 주소/좌표를 찾을 수 없음
   - 500 에러: 서버 내부 오류

## 📚 관련 문서

- [네이버 지도 API 공식 문서](https://api.ncloud-docs.com/docs/ai-naver-mapsgeocoding-geocoding)
- [네이버 지도 Reverse Geocoding API 문서](https://api.ncloud-docs.com/docs/ai-naver-mapsreversegeocoding-gc)

