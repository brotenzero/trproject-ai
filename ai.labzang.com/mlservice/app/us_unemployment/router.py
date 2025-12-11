"""
미국 실업률 데이터 시각화 관련 라우터
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Optional
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.us_unemployment.service import USUnemploymentService
from common.utils import create_response
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/usa", tags=["us_unemployment"])

# 서비스 인스턴스 (싱글톤)
_service_instance: Optional[USUnemploymentService] = None


def get_service() -> USUnemploymentService:
    """USUnemploymentService 싱글톤 인스턴스 반환"""
    global _service_instance
    if _service_instance is None:
        _service_instance = USUnemploymentService()
    return _service_instance


@router.get("/")
async def usa_root():
    """미국 실업률 서비스 루트"""
    return create_response(
        data={"service": "mlservice", "module": "us_unemployment", "status": "running"},
        message="US Unemployment Service is running",
    )


@router.get("/health")
async def health_check():
    """헬스 체크"""
    try:
        service = get_service()
        service.load_geo_data()
        service.load_unemployment_data()
        return create_response(
            data={"status": "healthy", "service": "us_unemployment"},
            message="US Unemployment service is healthy",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")


@router.get("/map", response_class=HTMLResponse)
async def get_unemployment_map(
    location_lat: Optional[float] = Query(48, description="지도 중심 위도"),
    location_lng: Optional[float] = Query(-102, description="지도 중심 경도"),
    zoom_start: Optional[int] = Query(3, description="지도 초기 줌 레벨"),
    fill_color: Optional[str] = Query("YlGn", description="색상 팔레트"),
    fill_opacity: Optional[float] = Query(0.7, description="채우기 투명도"),
    line_opacity: Optional[float] = Query(0.2, description="선 투명도"),
):
    """
    미국 실업률 데이터를 Choropleth 지도로 시각화하여 HTML로 반환합니다.
    
    Args:
        location_lat: 지도 중심 위도 (기본값: 48)
        location_lng: 지도 중심 경도 (기본값: -102)
        zoom_start: 지도 초기 줌 레벨 (기본값: 3)
        fill_color: 색상 팔레트 (기본값: "YlGn")
        fill_opacity: 채우기 투명도 (기본값: 0.7)
        line_opacity: 선 투명도 (기본값: 0.2)
    
    Returns:
        HTML 형식의 Folium 지도
    """
    try:
        # 커스텀 설정으로 서비스 생성
        service = USUnemploymentService(
            map_location=[location_lat, location_lng],
            zoom_start=zoom_start,
        )
        
        # 데이터 로드
        service.load_geo_data()
        service.load_unemployment_data()
        
        # 지도 생성
        service.create_map()
        
        # Choropleth 추가 (커스텀 설정 적용)
        service.add_choropleth(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
        )
        
        # 레이어 컨트롤 추가
        service.add_layer_control()
        
        # HTML로 변환
        if service.map is None:
            raise HTTPException(status_code=500, detail="지도 생성에 실패했습니다.")
        
        html_content = service.map._repr_html_()
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"지도 생성 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"지도 생성 중 오류: {str(e)}")


@router.get("/data")
async def get_unemployment_data():
    """
    미국 실업률 데이터를 JSON 형식으로 반환합니다.
    
    Returns:
        실업률 데이터 (State, Unemployment)
    """
    try:
        service = get_service()
        data = service.load_unemployment_data()
        
        data_clean = data.where(pd.notnull(data), None)
        
        return create_response(
            data={
                "rows": len(data),
                "columns": len(data.columns),
                "column_names": data.columns.tolist(),
                "data": data_clean.to_dict(orient="records"),
            },
            message="실업률 데이터 조회 완료",
        )
    except Exception as e:
        logger.error(f"데이터 로드 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"데이터 로드 중 오류: {str(e)}")

