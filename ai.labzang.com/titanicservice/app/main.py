"""
Titanic Service - FastAPI 애플리케이션
"""
import sys
import csv
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 공통 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import TitanicServiceConfig
from common.middleware import LoggingMiddleware
from common.utils import setup_logging

# 설정 로드
config = TitanicServiceConfig()

# 로깅 설정
logger = setup_logging(config.service_name)

# FastAPI 앱 생성
app = FastAPI(
    title="Titanic Service API",
    description="타이타닉 데이터 서비스 API 문서",
    version=config.service_version
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
app.add_middleware(LoggingMiddleware)

# CSV 파일 경로
CSV_FILE_PATH = Path(__file__).parent / "train.csv"


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
        "message": "Titanic Service API"
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

