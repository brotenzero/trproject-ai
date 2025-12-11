# Postman으로 CSV 전처리 확인 가이드

## 📋 개요
mlservice의 `/api/ml/seoul_crime/preprocess` 엔드포인트를 통해 CSV 데이터 전처리 결과를 확인할 수 있습니다.

## 🚀 빠른 시작

### 1. 기본 전처리 요청 (pop + cctv만, 범죄 데이터 제외)
**GET** `http://localhost:8080/api/ml/seoul_crime/preprocess`

**Query Parameters:**
- `pop_filename`: `pop.xls` (기본값)
- `cctv_filename`: `cctv.csv` (기본값)
- `crime_filename`: 생략 또는 `None` (기본값: 제외)
- `how`: `inner` (기본값) - 병합 방식: `inner` / `left` / `right` / `outer`

**예시:**
```
http://localhost:8080/api/ml/seoul_crime/preprocess
```
또는
```
http://localhost:8080/api/ml/seoul_crime/preprocess?pop_filename=pop.xls&cctv_filename=cctv.csv&how=inner
```

### 2. 범죄 데이터 포함 전처리 (Naver API 필요)
```
http://localhost:8080/api/ml/seoul_crime/preprocess?pop_filename=pop.xls&cctv_filename=cctv.csv&crime_filename=crime.csv&how=inner
```

**주의:** 범죄 데이터를 포함하려면 Naver API 인증 정보(`NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`)가 필요하며, Naver Developers에서 "Search" API 권한이 활성화되어 있어야 합니다.

## 📝 Postman 설정 방법

### Step 1: 새 Request 생성
1. Postman 열기
2. **New** → **HTTP Request** 선택
3. Method를 **GET**으로 설정

### Step 2: URL 입력
```
http://localhost:8080/api/ml/seoul_crime/preprocess
```

### Step 3: Query Parameters 설정
**Params** 탭에서 다음 파라미터 추가:

| Key | Value | Description |
|-----|-------|-------------|
| `pop_filename` | `pop.xls` | 인구 데이터 파일명 |
| `cctv_filename` | `cctv.csv` | CCTV 데이터 파일명 |
| `crime_filename` | `crime.csv` | 범죄 데이터 파일명 (선택) |
| `how` | `inner` | 병합 방식 |

### Step 4: Send 클릭
**Send** 버튼을 클릭하여 요청을 보냅니다.

## 📊 응답 예시

### 개별 DataFrame 응답 (pop, cctv, crime)

#### 인구 데이터 응답
```json
{
  "success": true,
  "message": "인구 데이터 조회 완료",
  "data": {
    "rows": 25,
    "columns": 2,
    "column_names": ["구", "인구합계"],
    "data": [
      {
        "구": "종로구",
        "인구합계": 156000
      },
      {
        "구": "중구",
        "인구합계": 134000
      },
      ...
    ]
  }
}
```

#### CCTV 데이터 응답
```json
{
  "success": true,
  "message": "CCTV 데이터 조회 완료",
  "data": {
    "rows": 25,
    "columns": 2,
    "column_names": ["기관명", "소계"],
    "data": [
      {
        "기관명": "종로구",
        "소계": 671
      },
      {
        "기관명": "중구",
        "소계": 884
      },
      ...
    ]
  }
}
```

#### 범죄 데이터 응답
```json
{
  "success": true,
  "message": "범죄 데이터 조회 완료",
  "data": {
    "rows": 31,
    "columns": 4,
    "column_names": ["관서명", "자치구", "검거합계", "나머지합계"],
    "data": [
      {
        "관서명": "중부서",
        "자치구": "중구",
        "검거합계": 1234,
        "나머지합계": 5678
      },
      ...
    ]
  }
}
```

### 병합 데이터 응답 (preprocess)

#### 성공 응답 (200 OK)
```json
{
  "success": true,
  "message": "병합 및 Top 5 조회 완료",
  "data": {
    "rows": 25,
    "columns": 10,
    "column_names": [
      "구",
      "인구합계",
      "소계",
      "검거합계",
      "나머지합계",
      ...
    ],
    "top5": [
      {
        "구": "강남구",
        "인구합계": 570500,
        "소계": 2780,
        "검거합계": 1234,
        "나머지합계": 5678
      },
      ...
    ]
  }
}
```

### 에러 응답 (404 Not Found)
```json
{
  "detail": "파일을 찾을 수 없습니다: ..."
}
```

## 🔍 다른 엔드포인트

### 개별 DataFrame 조회

#### 인구 데이터 (pop)
**GET** `http://localhost:8080/api/ml/seoul_crime/pop`

**Query Parameters:**
- `filename`: `pop.xls` (기본값)

**예시:**
```
http://localhost:8080/api/ml/seoul_crime/pop
http://localhost:8080/api/ml/seoul_crime/pop?filename=pop.xls
```

#### CCTV 데이터 (cctv)
**GET** `http://localhost:8080/api/ml/seoul_crime/cctv`

**Query Parameters:**
- `filename`: `cctv.csv` (기본값)

**예시:**
```
http://localhost:8080/api/ml/seoul_crime/cctv
http://localhost:8080/api/ml/seoul_crime/cctv?filename=cctv.csv
```

#### 범죄 데이터 (crime)
**GET** `http://localhost:8080/api/ml/seoul_crime/crime`

**Query Parameters:**
- `filename`: `crime.csv` (기본값)

**주의:** Naver API를 사용하여 관서명을 자치구로 변환하므로 시간이 걸릴 수 있습니다.

**예시:**
```
http://localhost:8080/api/ml/seoul_crime/crime
http://localhost:8080/api/ml/seoul_crime/crime?filename=crime.csv
```

### 병합된 데이터 조회

#### 전체 병합 데이터
**GET** `http://localhost:8080/api/ml/seoul_crime/preprocess/full`

**Query Parameters:**
- `pop_filename`: `pop.xls` (기본값)
- `cctv_filename`: `cctv.csv` (기본값)
- `crime_filename`: `crime.csv` (기본값, None이면 제외)
- `how`: `inner` (기본값) - 병합 방식

**예시:**
```
http://localhost:8080/api/ml/seoul_crime/preprocess/full
http://localhost:8080/api/ml/seoul_crime/preprocess/full?pop_filename=pop.xls&cctv_filename=cctv.csv&crime_filename=crime.csv&how=inner
```

### 기타 엔드포인트

#### Health Check
**GET** `http://localhost:8080/api/ml/seoul_crime/health`

#### 서비스 정보
**GET** `http://localhost:8080/api/ml/seoul_crime/`

#### Top 5만 조회
**GET** `http://localhost:8080/api/ml/seoul_crime/top5`

## ⚙️ 데이터 파일 위치
- `pop.xls`: `ai.labzang.com/mlservice/app/seoul_crime/data/pop.xls`
- `cctv.csv`: `ai.labzang.com/mlservice/app/seoul_crime/data/cctv.csv`
- `crime.csv`: `ai.labzang.com/mlservice/app/seoul_crime/data/crime.csv`

## 🐛 문제 해결

### 1. 404 Not Found
- Gateway와 mlservice가 실행 중인지 확인: `docker compose ps`
- URL 경로 확인: `/api/ml/seoul_crime/preprocess`

### 2. 500 Internal Server Error
- 로그 확인: `docker compose logs mlservice`
- 데이터 파일이 올바른 위치에 있는지 확인

### 3. Gateway Timeout (504)
- **현상**: Gateway에서 504 에러가 발생하지만, 실제로는 mlservice가 정상 응답(200 OK)을 반환하는 경우가 있습니다.
- **해결 방법**:
  - Postman에서 응답을 확인해보세요. 실제 데이터가 정상적으로 반환될 수 있습니다.
  - mlservice 로그 확인: `docker compose logs mlservice --tail 50`
  - Gateway timeout 설정 확인: `api.labzang.com/src/main/resources/application.yaml`의 `response-timeout` 값
  - crime 데이터를 사용하지 않는 경우, `crime_filename` 파라미터를 생략하세요.

## 📌 참고사항
- Gateway를 통해 요청하므로 포트는 `8080`을 사용합니다.
- mlservice 직접 접근 시 포트는 `9010`입니다.
- Naver API를 사용하는 경우 (crime 데이터 포함), 환경 변수 `NAVER_CLIENT_ID`와 `NAVER_CLIENT_SECRET`이 필요합니다.

