FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 한글 폰트 설치 (나눔고딕)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-nanum \
    fontconfig && \
    fc-cache -fv && \
    rm -rf /var/lib/apt/lists/*

# 공통 모듈 복사 (빌드 컨텍스트가 ai.labzang.com인 경우)
COPY common ./common

# requirements.txt 복사 및 설치
COPY mlservice/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY mlservice/app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9010"]

