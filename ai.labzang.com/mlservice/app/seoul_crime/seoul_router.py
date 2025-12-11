"""
서울 범죄 데이터 관련 라우터
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Optional
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.seoul_crime.seoul_service import SeoulService
from app.seoul_crime.seoul_naver_client import SeoulNaverClient
from common.utils import create_response
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/seoul_crime", tags=["seoul_crime"])

# 서비스 인스턴스 (싱글톤)
_service_instance: Optional[SeoulService] = None
_naver_client_instance: Optional[SeoulNaverClient] = None


def get_service() -> SeoulService:
    global _service_instance
    if _service_instance is None:
        _service_instance = SeoulService()
    return _service_instance


def get_naver_client() -> SeoulNaverClient:
    global _naver_client_instance
    if _naver_client_instance is None:
        _naver_client_instance = SeoulNaverClient()
    return _naver_client_instance


@router.get("/")
async def seoul_crime_root():
    return create_response(
        data={"service": "mlservice", "module": "seoul_crime", "status": "running"},
        message="Seoul Crime Service is running",
    )


@router.get("/health")
async def health_check():
    try:
        return create_response(
            data={"status": "healthy", "service": "seoul_crime"},
            message="Seoul Crime service is healthy",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")


@router.get("/top5")
async def get_top5(
    pop_filename: str = Query("pop.csv", description="인구 데이터 파일명"),
    cctv_filename: str = Query("cctv.csv", description="CCTV 데이터 파일명"),
    crime_filename: Optional[str] = Query("crime.csv", description="범죄 데이터 파일명 (None이면 제외)"),
    how: str = Query("inner", description="병합 방식: inner/left/right/outer"),
):
    """
    pop, cctv, crime을 '자치구' 기준으로 병합 후 상위 5행을 반환합니다.
    """
    try:
        service = get_service()
        merged = service.merge_all(
            pop_filename=pop_filename,
            cctv_filename=cctv_filename,
            crime_filename=crime_filename,
            how=how,
        )

        return create_response(
            data={
                "rows": len(merged),
                "columns": len(merged.columns),
                "column_names": merged.columns.tolist(),
                "top5": merged.head(5).to_dict(orient="records"),
            },
            message="병합 완료 (상위 5행)",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"병합 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"병합 중 오류: {str(e)}")


@router.get("/preprocess")
async def preprocess_and_top5(
    pop_filename: str = Query("pop.csv", description="인구 데이터 파일명"),
    cctv_filename: str = Query("cctv.csv", description="CCTV 데이터 파일명"),
    crime_filename: Optional[str] = Query(None, description="범죄 데이터 파일명 (생략하거나 None이면 제외)"),
    how: str = Query("inner", description="병합 방식: inner/left/right/outer"),
):
    """
    pop, cctv, crime을 '자치구' 기준으로 병합하고, Top 5를 바로 반환합니다.
    (Postman에서 /preprocess 호출만으로 확인)
    - crime 데이터는 Naver API를 통해 관서명을 자치구로 변환합니다.
    """
    try:
        service = get_service()
        merged = service.merge_all(
            pop_filename=pop_filename,
            cctv_filename=cctv_filename,
            crime_filename=crime_filename,
            how=how,
        )

        return create_response(
            data={
                "rows": len(merged),
                "columns": len(merged.columns),
                "column_names": merged.columns.tolist(),
                "top5": merged.head(5).to_dict(orient="records"),
            },
            message="병합 및 Top 5 조회 완료",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"병합 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"병합 중 오류: {str(e)}")


@router.get("/preprocess/full")
async def preprocess_full(
    pop_filename: Optional[str] = Query(None, description="인구 데이터 파일명 (기본값: pop.csv)"),
    cctv_filename: Optional[str] = Query(None, description="CCTV 데이터 파일명 (기본값: cctv.csv)"),
    crime_filename: Optional[str] = Query(None, description="범죄 데이터 파일명 (기본값: crime.csv, None이면 제외)"),
    how: Optional[str] = Query(None, description="병합 방식: inner/left/right/outer (기본값: inner)"),
):
    """
    pop, cctv, crime을 '자치구' 기준으로 병합하고, 전처리된 전체 DataFrame을 반환합니다.
    (Postman에서 전체 데이터 확인용)
    - crime 데이터는 Naver API를 통해 관서명을 자치구로 변환합니다.
    - 전체 데이터를 JSON 형태로 반환합니다.
    """
    try:
        service = get_service()
        merged = service.merge_all(
            pop_filename=pop_filename or "pop.csv",
            cctv_filename=cctv_filename or "cctv.csv",
            crime_filename=crime_filename or "crime.csv",
            how=how or "inner",
        )

        # NaN 값을 None으로 변환하여 JSON 직렬화 가능하게 함
        merged_clean = merged.where(pd.notnull(merged), None)

        return create_response(
            data={
                "rows": len(merged),
                "columns": len(merged.columns),
                "column_names": merged.columns.tolist(),
                "data": merged_clean.to_dict(orient="records"),
            },
            message="전처리된 전체 데이터 조회 완료",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"병합 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"병합 중 오류: {str(e)}")


@router.get("/pop")
async def get_pop_data(
    filename: str = Query("pop.csv", description="인구 데이터 파일명"),
):
    """
    인구 데이터를 로드하여 DataFrame을 반환합니다.
    """
    try:
        service = get_service()
        pop_df = service.load_pop(filename)
        
        # NaN 값을 None으로 변환하여 JSON 직렬화 가능하게 함
        pop_clean = pop_df.where(pd.notnull(pop_df), None)
        
        return create_response(
            data={
                "rows": len(pop_df),
                "columns": len(pop_df.columns),
                "column_names": pop_df.columns.tolist(),
                "data": pop_clean.to_dict(orient="records"),
            },
            message="인구 데이터 조회 완료",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"인구 데이터 로드 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"인구 데이터 로드 중 오류: {str(e)}")


@router.get("/cctv")
async def get_cctv_data(
    filename: str = Query("cctv.csv", description="CCTV 데이터 파일명"),
):
    """
    CCTV 데이터를 로드하여 DataFrame을 반환합니다.
    """
    try:
        service = get_service()
        cctv_df = service.load_cctv(filename)
        
        # NaN 값을 None으로 변환하여 JSON 직렬화 가능하게 함
        cctv_clean = cctv_df.where(pd.notnull(cctv_df), None)
        
        return create_response(
            data={
                "rows": len(cctv_df),
                "columns": len(cctv_df.columns),
                "column_names": cctv_df.columns.tolist(),
                "data": cctv_clean.to_dict(orient="records"),
            },
            message="CCTV 데이터 조회 완료",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"CCTV 데이터 로드 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CCTV 데이터 로드 중 오류: {str(e)}")


@router.get("/crime")
async def get_crime_data(
    filename: str = Query("crime.csv", description="범죄 데이터 파일명"),
    pop_filename: str = Query("pop.csv", description="인구 데이터 파일명"),
):
    """
    범죄 데이터를 로드하여 자치구별로 합계를 내고 각 범죄별 범죄율(10만 명당 발생 건수)을 계산하여 DataFrame을 반환합니다.
    - Naver API를 통해 관서명을 자치구로 변환합니다.
    - 같은 구에 여러 관서가 있으면 합계를 계산합니다.
    - 각 범죄별 범죄율(살인범죄율, 강도범죄율, 강간범죄율, 절도범죄율, 폭력범죄율)을 계산합니다.
    """
    try:
        service = get_service()
        crime_df = service.load_crime(filename, pop_filename)
        
        # NaN 값을 None으로 변환하여 JSON 직렬화 가능하게 함
        crime_clean = crime_df.where(pd.notnull(crime_df), None)
        
        return create_response(
            data={
                "rows": len(crime_df),
                "columns": len(crime_df.columns),
                "column_names": crime_df.columns.tolist(),
                "data": crime_clean.to_dict(orient="records"),
            },
            message="범죄 데이터 조회 완료",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"범죄 데이터 로드 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"범죄 데이터 로드 중 오류: {str(e)}")


@router.get("/crime/arrest-rate")
async def get_crime_arrest_rate_data(
    filename: str = Query("crime.csv", description="범죄 데이터 파일명"),
    pop_filename: str = Query("pop.csv", description="인구 데이터 파일명"),
):
    """
    범죄 데이터를 로드하여 자치구별로 합계를 내고 각 범죄별 범죄율(10만 명당 발생 건수)을 계산하여 DataFrame을 반환합니다.
    - Naver API를 통해 관서명을 자치구로 변환합니다.
    - 같은 구에 여러 관서가 있으면 합계를 계산합니다.
    - 각 범죄별 범죄율(살인범죄율, 강도범죄율, 강간범죄율, 절도범죄율, 폭력범죄율)을 계산합니다.
    """
    try:
        service = get_service()
        crime_df = service.load_crime(filename, pop_filename)
        
        # NaN 값을 None으로 변환하여 JSON 직렬화 가능하게 함
        crime_clean = crime_df.where(pd.notnull(crime_df), None)
        
        return create_response(
            data={
                "rows": len(crime_df),
                "columns": len(crime_df.columns),
                "column_names": crime_df.columns.tolist(),
                "data": crime_clean.to_dict(orient="records"),
            },
            message="범죄율 데이터 조회 완료",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"범죄 검거율 데이터 로드 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"범죄 검거율 데이터 로드 중 오류: {str(e)}")


@router.get("/geocode")
async def geocode_address(
    query: str = Query(..., description="검색할 주소 (예: 서울특별시 강남구 테헤란로 152)"),
):
    """
    네이버 지도 Geocoding API: 주소를 좌표로 변환
    
    주소를 입력하면 위도, 경도 좌표를 반환합니다.
    """
    try:
        naver_client = get_naver_client()
        result = naver_client.geocode(query)
        
        if result:
            return create_response(
                data=result,
                message="Geocoding 성공",
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"주소를 찾을 수 없습니다: {query}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Geocoding 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Geocoding 중 오류: {str(e)}")


@router.get("/reverse-geocode")
async def reverse_geocode_coords(
    latitude: float = Query(..., description="위도"),
    longitude: float = Query(..., description="경도"),
):
    """
    네이버 지도 Reverse Geocoding API: 좌표를 주소로 변환
    
    위도, 경도를 입력하면 주소를 반환합니다.
    """
    try:
        naver_client = get_naver_client()
        result = naver_client.reverse_geocode(latitude, longitude)
        
        if result:
            return create_response(
                data=result,
                message="Reverse Geocoding 성공",
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"좌표에 해당하는 주소를 찾을 수 없습니다: 위도={latitude}, 경도={longitude}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reverse Geocoding 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reverse Geocoding 중 오류: {str(e)}")


@router.get("/heatmap", response_class=HTMLResponse)
async def get_heatmap(
    pop_filename: str = Query("pop.csv", description="인구 데이터 파일명"),
    cctv_filename: str = Query("cctv.csv", description="CCTV 데이터 파일명"),
    crime_filename: Optional[str] = Query("crime.csv", description="범죄 데이터 파일명 (None이면 제외)"),
    how: str = Query("inner", description="병합 방식: inner/left/right/outer"),
    cmap: str = Query("YlOrRd", description="색상 맵 (예: YlOrRd, RdYlBu, viridis)"),
    annot: bool = Query(True, description="셀에 값 표시 여부"),
):
    """
    병합된 DataFrame의 범죄율 데이터를 히트맵으로 시각화하여 HTML로 반환합니다.
    - 자치구별 범죄율(살인, 강도, 강간, 절도, 폭력)을 히트맵으로 표시합니다.
    """
    try:
        service = get_service()
        img_base64 = service.create_heatmap(
            pop_filename=pop_filename,
            cctv_filename=cctv_filename,
            crime_filename=crime_filename,
            how=how,
            cmap=cmap,
            annot=annot,
        )
        
        # HTML 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>서울시 자치구별 범죄율 히트맵</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .info {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 4px;
                    font-size: 14px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>서울시 자치구별 범죄율 히트맵</h1>
                <div class="image-container">
                    <img src="data:image/png;base64,{img_base64}" alt="범죄율 히트맵" />
                </div>
                <div class="info">
                    <p><strong>설명:</strong></p>
                    <ul>
                        <li>각 셀의 값은 각 범죄 유형별로 최소 0에서 최대 1로 정규화된 값을 나타냅니다. (상대적 비교용)</li>
                        <li>색상이 진할수록 범죄율이 높습니다.</li>
                        <li>범죄 유형: 살인, 강도, 강간, 절도, 폭력</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"히트맵 생성 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"히트맵 생성 중 오류: {str(e)}")


@router.get("/heatmap/arrest-rate", response_class=HTMLResponse)
async def get_arrest_rate_heatmap(
    filename: str = Query("crime.csv", description="범죄 데이터 파일명"),
    cmap: str = Query("RdYlGn", description="색상 맵 (예: RdYlGn, YlGn, viridis)"),
    annot: bool = Query(True, description="셀에 값 표시 여부"),
):
    """
    검거율 데이터를 히트맵으로 시각화하여 HTML로 반환합니다.
    - 자치구별 범죄 검거율(살인, 강도, 강간, 절도, 폭력)을 히트맵으로 표시합니다.
    - 검거율은 0~100% 범위로 표시됩니다.
    """
    try:
        service = get_service()
        img_base64 = service.create_arrest_rate_heatmap(
            filename=filename,
            cmap=cmap,
            annot=annot,
        )
        
        # HTML 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>서울시 자치구별 범죄 검거율 히트맵</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .info {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 4px;
                    font-size: 14px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>서울시 자치구별 범죄 검거율 히트맵</h1>
                <div class="image-container">
                    <img src="data:image/png;base64,{img_base64}" alt="검거율 히트맵" />
                </div>
                <div class="info">
                    <p><strong>설명:</strong></p>
                    <ul>
                        <li>각 셀의 값은 검거율(%)을 나타냅니다.</li>
                        <li>검거율 = (검거 건수 / 발생 건수) × 100</li>
                        <li>색상이 진할수록(초록색) 검거율이 높고, 연할수록(빨강색) 검거율이 낮습니다.</li>
                        <li>범죄 유형: 살인, 강도, 강간, 절도, 폭력</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"검거율 히트맵 생성 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"검거율 히트맵 생성 중 오류: {str(e)}")


@router.get("/map", response_class=HTMLResponse)
async def get_crime_map(
    crime_filename: str = Query("crime.csv", description="범죄 데이터 파일명"),
    pop_filename: str = Query("pop.csv", description="인구 데이터 파일명"),
    location_lat: Optional[float] = Query(37.5665, description="지도 중심 위도"),
    location_lng: Optional[float] = Query(126.9780, description="지도 중심 경도"),
    zoom_start: Optional[int] = Query(11, description="지도 초기 줌 레벨"),
    crime_rate_fill_color: Optional[str] = Query("YlOrRd", description="범죄율 색상 팔레트"),
    arrest_rate_fill_color: Optional[str] = Query("RdYlGn", description="검거율 색상 팔레트"),
    fill_opacity: Optional[float] = Query(0.7, description="채우기 투명도"),
    line_opacity: Optional[float] = Query(0.2, description="선 투명도"),
):
    """
    서울시 자치구별 범죄율과 검거율을 지도로 시각화하여 HTML로 반환합니다.
    - 범죄율과 검거율은 서로 다른 레이어로 표시됩니다.
    - 레이어 컨트롤을 통해 범죄율/검거율 레이어를 전환할 수 있습니다.
    """
    try:
        service = get_service()
        m = service.create_crime_map(
            crime_filename=crime_filename,
            pop_filename=pop_filename,
            map_location=[location_lat, location_lng],
            zoom_start=zoom_start,
            crime_rate_fill_color=crime_rate_fill_color,
            arrest_rate_fill_color=arrest_rate_fill_color,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
        )
        
        # HTML로 변환
        html_content = m._repr_html_()
        
        return HTMLResponse(content=html_content)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"지도 생성 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"지도 생성 중 오류: {str(e)}")