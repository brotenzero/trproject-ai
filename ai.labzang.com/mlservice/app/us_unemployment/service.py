"""
미국 실업률 데이터 시각화 서비스
"""

from typing import Optional, Dict, Any
import logging

import pandas as pd
import requests
import folium

logger = logging.getLogger(__name__)


class USUnemploymentService:
    """
    미국 실업률 데이터를 로드하고 지도에 시각화하는 서비스
    """

    def __init__(
        self,
        geo_data_url: str = "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json",
        data_url: str = "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_unemployment_oct_2012.csv",
        map_location: list = [48, -102],
        zoom_start: int = 3,
    ):
        """
        서비스 초기화
        
        Args:
            geo_data_url: 지리 데이터 JSON URL
            data_url: 실업률 데이터 CSV URL
            map_location: 지도 중심 좌표 [위도, 경도]
            zoom_start: 지도 초기 줌 레벨
        """
        self.geo_data_url = geo_data_url
        self.data_url = data_url
        self.map_location = map_location
        self.zoom_start = zoom_start
        
        self.state_geo: Optional[Dict[str, Any]] = None
        self.state_data: Optional[pd.DataFrame] = None
        self.map: Optional[folium.Map] = None

    def load_geo_data(self) -> Dict[str, Any]:
        """
        지리 데이터(GeoJSON)를 로드합니다.
        
        Returns:
            지리 데이터 딕셔너리
        """
        if self.state_geo is None:
            try:
                logger.info(f"지리 데이터 로드 중: {self.geo_data_url}")
                response = requests.get(self.geo_data_url, timeout=10)
                response.raise_for_status()
                self.state_geo = response.json()
                logger.info("지리 데이터 로드 완료")
            except requests.exceptions.RequestException as e:
                logger.error(f"지리 데이터 로드 실패: {e}")
                raise
        
        return self.state_geo

    def load_unemployment_data(self) -> pd.DataFrame:
        """
        실업률 데이터를 로드합니다.
        
        Returns:
            실업률 데이터 DataFrame
        """
        if self.state_data is None:
            try:
                logger.info(f"실업률 데이터 로드 중: {self.data_url}")
                self.state_data = pd.read_csv(self.data_url)
                logger.info(f"실업률 데이터 로드 완료: {len(self.state_data)}행")
            except Exception as e:
                logger.error(f"실업률 데이터 로드 실패: {e}")
                raise
        
        return self.state_data

    def create_map(
        self,
        location: Optional[list] = None,
        zoom_start: Optional[int] = None,
    ) -> folium.Map:
        """
        Folium 지도를 생성합니다.
        
        Args:
            location: 지도 중심 좌표 [위도, 경도] (기본값: self.map_location)
            zoom_start: 지도 초기 줌 레벨 (기본값: self.zoom_start)
        
        Returns:
            Folium Map 객체
        """
        if location is None:
            location = self.map_location
        if zoom_start is None:
            zoom_start = self.zoom_start
        
        logger.info(f"지도 생성 중: location={location}, zoom_start={zoom_start}")
        self.map = folium.Map(location=location, zoom_start=zoom_start)
        logger.info("지도 생성 완료")
        
        return self.map

    def add_choropleth(
        self,
        fill_color: str = "YlGn",
        fill_opacity: float = 0.7,
        line_opacity: float = 0.2,
        legend_name: str = "Unemployment Rate (%)",
    ) -> folium.Choropleth:
        """
        Choropleth 레이어를 지도에 추가합니다.
        
        Args:
            fill_color: 색상 팔레트 (기본값: "YlGn")
            fill_opacity: 채우기 투명도 (기본값: 0.7)
            line_opacity: 선 투명도 (기본값: 0.2)
            legend_name: 범례 이름 (기본값: "Unemployment Rate (%)")
        
        Returns:
            Choropleth 객체
        """
        if self.map is None:
            self.create_map()
        
        if self.state_geo is None:
            self.load_geo_data()
        
        if self.state_data is None:
            self.load_unemployment_data()
        
        logger.info("Choropleth 레이어 추가 중")
        choropleth = folium.Choropleth(
            geo_data=self.state_geo,
            name="choropleth",
            data=self.state_data,
            columns=["State", "Unemployment"],
            key_on="feature.id",
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
            legend_name=legend_name,
        )
        choropleth.add_to(self.map)
        logger.info("Choropleth 레이어 추가 완료")
        
        return choropleth

    def add_layer_control(self) -> folium.LayerControl:
        """
        레이어 컨트롤을 지도에 추가합니다.
        
        Returns:
            LayerControl 객체
        """
        if self.map is None:
            raise ValueError("지도가 생성되지 않았습니다. create_map()을 먼저 호출하세요.")
        
        logger.info("레이어 컨트롤 추가 중")
        layer_control = folium.LayerControl()
        layer_control.add_to(self.map)
        logger.info("레이어 컨트롤 추가 완료")
        
        return layer_control

    def get_map(self) -> folium.Map:
        """
        완성된 지도를 반환합니다.
        데이터를 로드하고 Choropleth와 레이어 컨트롤을 자동으로 추가합니다.
        
        Returns:
            완성된 Folium Map 객체
        """
        # 데이터 로드
        self.load_geo_data()
        self.load_unemployment_data()
        
        # 지도 생성
        self.create_map()
        
        # Choropleth 추가
        self.add_choropleth()
        
        # 레이어 컨트롤 추가
        self.add_layer_control()
        
        return self.map
