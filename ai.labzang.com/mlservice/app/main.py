"""
Titanic Service - FastAPI 애플리케이션
"""
import sys
import csv
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 공통 모듈 경로 추가 (최우선)
current_file = Path(__file__).resolve()
base_dir = current_file.parent.parent  # /app (Docker) 또는 mlservice (로컬)

# 경로 추가
base_path_str = str(base_dir)
if base_path_str not in sys.path:
    sys.path.insert(0, base_path_str)

# Docker 환경 확인 및 /app 경로 추가
if os.path.exists("/app"):
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

# 설정 로드 (경로 설정 후)
try:
    from app.config import TitanicServiceConfig
    config = TitanicServiceConfig()
except Exception as e:
    # config.py를 찾을 수 없는 경우 기본값 사용
    class Config:
        service_name = "mlservice"
        service_version = "1.0.0"
        port = 9010
    config = Config()

# 라우터 및 공통 모듈 import
LoggingMiddleware = None
try:
    from app.titanic.titanic_router import router as titanic_router
    from app.seoul_crime.seoul_router import router as seoul_router
    from app.us_unemployment.router import router as usa_router
    from app.nlp.nlp_router import router as nlp_router
    from common.middleware import LoggingMiddleware
    from common.utils import setup_logging
except ImportError as e:
    # 모듈을 찾을 수 없는 경우 기본값 사용
    from fastapi import APIRouter
    titanic_router = APIRouter()
    seoul_router = APIRouter()
    usa_router = APIRouter()
    nlp_router = APIRouter()
    def setup_logging(name):
        import logging
        return logging.getLogger(name)

# 로깅 설정
logger = setup_logging(config.service_name)

# FastAPI 앱 생성
app = FastAPI(
    title="ML Service API",
    description="""
    ## ML 서비스 통합 API
    
    머신러닝과 자연어 처리를 활용한 다양한 데이터 분석 및 예측 서비스입니다.
    
    ### 주요 기능
    
    #### 1. 타이타닉 데이터 분석
    - 승객 데이터 조회 및 통계 분석
    - 머신러닝 모델 훈련 (Random Forest)
    - 승객 생존 예측
    
    #### 2. 서울 범죄 데이터 분석
    - 범죄 데이터 전처리 및 시각화
    - 지역별 범죄율 분석
    - 히트맵 및 지도 시각화
    
    #### 3. 미국 실업률 데이터
    - 실업률 데이터 시각화
    
    #### 4. 자연어 처리 (NLP) ✨
    - NLTK 기반 텍스트 분석
    - 토큰화, 형태소 분석, 품사 태깅
    - 워드클라우드 생성
    - 말뭉치(Corpus) 분석
    
    ### 기술 스택
    - **Framework**: FastAPI
    - **ML Library**: scikit-learn, pandas, numpy
    - **NLP Library**: NLTK, WordCloud
    - **Visualization**: matplotlib, seaborn, folium
    
    ### API 문서
    - Swagger UI: `/docs`
    - ReDoc: `/redoc`
    - OpenAPI Schema: `/openapi.json`
    """,
    version=config.service_version,
    contact={
        "name": "ML Service Team",
        "email": "support@labzang.com",
    },
    license_info={
        "name": "MIT",
    },
    tags_metadata=[
        {
            "name": "titanic",
            "description": "타이타닉 승객 데이터 관련 API",
        },
        {
            "name": "seoul_crime",
            "description": "서울 범죄 데이터 전처리 및 분석 API",
        },
        {
            "name": "us_unemployment",
            "description": "미국 실업률 데이터 시각화 API",
        },
        {
            "name": "nlp",
            "description": "자연어 처리(NLP) 및 텍스트 분석 API",
        },
    ],
    openapi_tags=[
        {
            "name": "titanic",
            "description": "타이타닉 승객 데이터 및 머신러닝 예측 기능",
        },
        {
            "name": "seoul_crime",
            "description": "서울 범죄 데이터 전처리 및 분석 기능",
        },
        {
            "name": "us_unemployment",
            "description": "미국 실업률 데이터 시각화 기능",
        },
        {
            "name": "nlp",
            "description": "NLTK 기반 자연어 처리, 토큰화, 형태소 분석, 품사 태깅, 워드클라우드 생성",
        },
    ],
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 미들웨어 추가
if LoggingMiddleware is not None:
    app.add_middleware(LoggingMiddleware)

# 라우터 등록
# Gateway에서 이미 /ml로 rewrite하므로 prefix 없이 등록
app.include_router(titanic_router, prefix="/ml")
app.include_router(seoul_router, prefix="/ml")
app.include_router(usa_router, prefix="/ml")
app.include_router(nlp_router, prefix="/ml")

# CSV 파일 경로
CSV_FILE_PATH = Path(__file__).parent / "resources" / "titanic" / "train.csv"


def load_top_10_passengers():
    """train.csv에서 상위 10명의 승객 정보를 로드"""
    passengers = []
    
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 10:  # 상위 10명만
                    break
                passengers.append({
                    "PassengerId": row.get("PassengerId", ""),
                    "Survived": row.get("Survived", ""),
                    "Pclass": row.get("Pclass", ""),
                    "Name": row.get("Name", ""),
                    "Sex": row.get("Sex", ""),
                    "Age": row.get("Age", ""),
                    "SibSp": row.get("SibSp", ""),
                    "Parch": row.get("Parch", ""),
                    "Ticket": row.get("Ticket", ""),
                    "Fare": row.get("Fare", ""),
                    "Cabin": row.get("Cabin", ""),
                    "Embarked": row.get("Embarked", "")
                })
    except FileNotFoundError:
        logger.error(f"CSV 파일을 찾을 수 없습니다: {CSV_FILE_PATH}")
        return []
    except Exception as e:
        logger.error(f"CSV 파일 읽기 오류: {e}")
        return []
    
    return passengers


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": config.service_name,
        "version": config.service_version,
        "message": "ML Service API - Machine Learning & NLP Services"
    }


@app.get("/passengers/top10")
async def get_top_10_passengers():
    """상위 10명의 승객 정보를 반환"""
    passengers = load_top_10_passengers()
    
    if not passengers:
        return JSONResponse(
            status_code=404,
            content={"error": "승객 데이터를 찾을 수 없습니다."}
        )
    
    return {
        "count": len(passengers),
        "passengers": passengers
    }


@app.get("/passengers/top10/print")
async def print_top_10_passengers():
    """상위 10명의 승객 정보를 터미널에 출력"""
    passengers = load_top_10_passengers()
    
    if not passengers:
        logger.warning("출력할 승객 데이터가 없습니다.")
        return {"message": "출력할 승객 데이터가 없습니다."}
    
    # 터미널에 출력
    print("\n" + "="*80)
    print("타이타닉 승객 상위 10명")
    print("="*80)
    
    for i, passenger in enumerate(passengers, 1):
        print(f"\n[{i}] {passenger['Name']}")
        print(f"    PassengerId: {passenger['PassengerId']}")
        print(f"    Survived: {passenger['Survived']} ({'생존' if passenger['Survived'] == '1' else '사망'})")
        print(f"    Pclass: {passenger['Pclass']}")
        print(f"    Sex: {passenger['Sex']}")
        print(f"    Age: {passenger['Age']}")
        print(f"    Fare: {passenger['Fare']}")
        print(f"    Embarked: {passenger['Embarked']}")
    
    print("\n" + "="*80)
    logger.info(f"상위 10명의 승객 정보를 터미널에 출력했습니다.")
    
    return {
        "message": "상위 10명의 승객 정보를 터미널에 출력했습니다.",
        "count": len(passengers)
    }


@app.on_event("startup")
async def startup_event():
    """서비스 시작 시 실행"""
    logger.info(f"{config.service_name} v{config.service_version} started")
    # 시작 시 상위 10명 출력
    await print_top_10_passengers()


@app.on_event("shutdown")
async def shutdown_event():
    """서비스 종료 시 실행"""
    logger.info(f"{config.service_name} shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.port)

