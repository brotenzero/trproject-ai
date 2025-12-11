"""
서울 범죄 데이터 서비스
"""

from pathlib import Path
from typing import Dict, Optional
import logging
import io
import base64
import os

import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 없이 사용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import folium

from app.seoul_crime.seoul_naver_client import SeoulNaverClient

logger = logging.getLogger(__name__)


class SeoulService:
    """
    pop(자치구)와 cctv(기관명)를 공통 키 '구'로 병합하는 서비스
    """

    def __init__(self):
        self.the_method = None
        self.dataset = None
        self.crime_rate_columns = ["살인검거율", "강도검거율", "강간검거율", "절도검거율", "폭력검거율"]
        self.crime_columns = ["살인", "강도", "강간", "절도", "폭력"]

        current_file = Path(__file__).resolve()
        self.data_dir = current_file.parent / "data"
        
        # 한글 폰트 설정
        self._setup_korean_font()
        
        # Naver API 클라이언트 초기화
        logger.info("[SeoulService] Naver API 클라이언트 초기화 시작")
        self.naver_client = SeoulNaverClient()
        logger.info("[SeoulService] Naver API 클라이언트 초기화 완료")
    
    def _setup_korean_font(self):
        """한글 폰트 설정"""
        try:
            # 시스템에 설치된 한글 폰트 찾기
            font_list = ['NanumGothic', 'Nanum Gothic', 'NanumGothicOTF', 
                        'Malgun Gothic', 'AppleGothic', 'Noto Sans CJK KR']
            
            # matplotlib 폰트 매니저에서 사용 가능한 폰트 확인
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            font_found = False
            for font_name in font_list:
                if font_name in available_fonts:
                    plt.rcParams['font.family'] = font_name
                    logger.info(f"한글 폰트 설정 완료: {font_name}")
                    font_found = True
                    break
            
            if not font_found:
                # 폰트를 찾을 수 없으면 기본 폰트 사용
                plt.rcParams['font.family'] = 'DejaVu Sans'
                logger.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
        except Exception as e:
            logger.warning(f"한글 폰트 설정 중 오류: {e}, 기본 폰트 사용")
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
        plt.rcParams['axes.unicode_minus'] = False

    def load_cctv(self, filename: str = "cctv.csv") -> pd.DataFrame:
        """
        cctv.csv 파일에서 기관명(자치구)과 소계만 추출하여 DataFrame으로 반환
        
        Args:
            filename: cctv 데이터 파일명 (기본값: cctv.csv)
            
        Returns:
            기관명과 소계만 포함된 DataFrame
        """
        cctv_path = self.data_dir / filename
        
        # CSV 파일 읽기
        df = pd.read_csv(cctv_path)
        
        # 컬럼 이름 정리 (따옴표 제거)
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # '기관명'과 '소계' 컬럼만 선택
        if "기관명" in df.columns and "소계" in df.columns:
            df = df[["기관명", "소계"]].copy()
        else:
            raise ValueError(f"필수 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
        
        logger.info(f"CCTV 데이터 로드 완료: {len(df)}행")
        
        return df

    def load_pop(self, filename: str = "pop.csv") -> pd.DataFrame:
        """
        pop.csv 파일에서 첫 두 행을 삭제하고 2열과 4열만 추출하여 DataFrame으로 반환
        
        Args:
            filename: 인구 데이터 파일명 (기본값: pop.csv)
            
        Returns:
            자치구(구)와 인구합계만 포함된 DataFrame
        """
        pop_path = self.data_dir / filename
        
        # CSV 파일 읽기 (첫 두 행 건너뛰기, 3행을 헤더로 사용)
        try:
            # skiprows=[0, 1]로 첫 두 행 건너뛰기, header=0으로 3행(이제 첫 번째 행)을 컬럼명으로 사용
            df = pd.read_csv(pop_path, encoding='utf-8-sig', skiprows=[0, 1], header=0)
        except Exception as e:
            logger.error(f"CSV 파일 읽기 실패: {e}")
            # 대안: header=None으로 읽고 수동 처리
            df = pd.read_csv(pop_path, encoding='utf-8-sig', header=None)
            # 첫 두 행 삭제 (인덱스 0, 1)
            df = df.drop([0, 1]).reset_index(drop=True)
            # 3행(인덱스 0)을 컬럼명으로 사용
            df.columns = df.iloc[0]
            df = df.drop(0).reset_index(drop=True)
        
        # 불필요한 빈 행 제거
        df = df.dropna(how='all').copy()
        
        # 2열(인덱스 1)과 4열(인덱스 3)만 선택하여 DataFrame 생성
        cols_list = list(df.columns)
        if len(cols_list) > 3:
            두번째_컬럼 = cols_list[1]  # 2열 (인덱스 1)
            네번째_컬럼 = cols_list[3]  # 4열 (인덱스 3)
            logger.info(f"2열 선택: {두번째_컬럼}, 4열 선택: {네번째_컬럼}")
            
            # 2열과 4열만 선택하여 DataFrame 생성
            df = df[[두번째_컬럼, 네번째_컬럼]].copy()
            # 컬럼명 변경: 2열을 '구'로, 4열을 '인구합계'로
            df = df.rename(columns={두번째_컬럼: "구", 네번째_컬럼: "인구합계"})
        else:
            raise ValueError(f"컬럼이 부족합니다. 전체 컬럼: {cols_list}")
        
        # 불필요한 합계/소계 행 제거
        df = df[df["구"].astype(str).str.strip() != "소계"].copy()
        df = df[df["구"].astype(str).str.strip() != "합계"].copy()
        df = df[df["구"].astype(str).str.strip() != ""].copy()
        df = df[df["구"].notna()].copy()
        
        logger.info(f"인구 데이터 로드 완료: {len(df)}행")
        
        return df

    def load_crime(self, filename: str = "crime.csv", pop_filename: str = "pop.csv") -> pd.DataFrame:
        """
        crime.csv 파일을 읽어서 관서명을 Naver API로 자치구를 찾고,
        같은 구 단위로 합계를 내서 각 범죄별 범죄율(10만 명당 발생 건수)을 계산하여 DataFrame으로 반환
        
        Args:
            filename: 범죄 데이터 파일명 (기본값: crime.csv)
            pop_filename: 인구 데이터 파일명 (기본값: pop.csv)
            
        Returns:
            자치구, 관서명(쉼표 구분), 각 범죄별 범죄율을 포함한 DataFrame
            컬럼: 자치구, 관서명, 살인범죄율, 강도범죄율, 강간범죄율, 절도범죄율, 폭력범죄율
        """
        crime_path = self.data_dir / filename
        
        # CSV 파일 읽기
        df = pd.read_csv(crime_path)
        
        logger.info(f"[load_crime] 범죄 데이터 로드: {len(df)}행, 컬럼: {list(df.columns)}")
        
        # 관서명 컬럼 확인
        if "관서명" not in df.columns:
            raise ValueError(f"'관서명' 컬럼을 찾을 수 없습니다. 컬럼: {list(df.columns)}")
        
        # 각 관서명에 대해 Naver API로 자치구 조회
        districts = []
        
        for station_name in df["관서명"]:
            logger.info(f"[load_crime] 관서명 조회 시작: {station_name}")
            try:
                station_info = self.naver_client.search_police_station(station_name)
                logger.info(f"[load_crime] Naver API 응답 받음: {station_name}, 결과: {station_info is not None}")
            except Exception as e:
                logger.error(f"[load_crime] Naver API 호출 중 오류 발생: {station_name}, 오류: {str(e)}", exc_info=True)
                station_info = None
            
            if station_info:
                district = station_info.get("district", "")
                logger.info(f"[load_crime] 관서: {station_name} | 자치구: {district}")
                districts.append(district)
            else:
                logger.warning(f"관서 정보를 찾을 수 없습니다: {station_name}")
                districts.append(None)
        
        # 자치구 컬럼 추가
        df["자치구"] = districts
        
        # 숫자 컬럼 정리 (쉼표 제거 및 숫자 변환)
        numeric_columns = df.select_dtypes(include=["object"]).columns
        for col in numeric_columns:
            if col not in ["관서명", "자치구"]:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "")
                    .str.replace('"', "")
                    .str.replace("-", "0")
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 결측치 처리 (관서명, 자치구는 제외)
        numeric_cols_for_fillna = [c for c in df.columns if c not in ["관서명", "자치구"]]
        df[numeric_cols_for_fillna] = df[numeric_cols_for_fillna].fillna(0)
        
        # 자치구별로 그룹화하여 합계 계산
        # 발생 컬럼만 사용 (범죄율 계산용)
        발생_컬럼들 = ["살인 발생", "강도 발생", "강간 발생", "절도 발생", "폭력 발생"]
        
        # 자치구별로 집계 (발생은 합계로)
        agg_dict = {}
        for col in 발생_컬럼들:
            if col in df.columns:
                agg_dict[col] = "sum"
        
        # 관서명은 쉼표로 구분된 문자열로
        def join_stations(x):
            unique_stations = x.astype(str).unique()
            return ", ".join([s for s in unique_stations if s and s != 'nan'])
        agg_dict["관서명"] = join_stations
        
        # 자치구가 None인 행 제외
        df_with_district = df[df["자치구"].notna()].copy()
        
        # 자치구별로 집계
        grouped_df = df_with_district.groupby("자치구").agg(agg_dict).reset_index()
        
        # 인구 데이터 로드
        pop_df = self.load_pop(pop_filename)
        # 인구 데이터의 '구' 컬럼을 '자치구'로 변환하여 병합
        pop_df = pop_df.rename(columns={"구": "자치구"})
        
        # 자치구별 범죄 데이터와 인구 데이터 병합
        grouped_df = pd.merge(grouped_df, pop_df[["자치구", "인구합계"]], on="자치구", how="left")
        
        # 인구가 없는 경우 0으로 처리
        grouped_df["인구합계"] = grouped_df["인구합계"].fillna(0)
        
        # 각 범죄별 범죄율 계산 (10만 명당 발생 건수)
        crime_rate_mapping = {
            "살인범죄율": "살인 발생",
            "강도범죄율": "강도 발생",
            "강간범죄율": "강간 발생",
            "절도범죄율": "절도 발생",
            "폭력범죄율": "폭력 발생",
        }
        
        for rate_col, 발생_col in crime_rate_mapping.items():
            if 발생_col in grouped_df.columns:
                # 범죄율 = (발생 건수 / 인구) * 100,000 (10만 명당)
                # 인구가 0이면 범죄율도 0으로 처리
                grouped_df[rate_col] = grouped_df.apply(
                    lambda row: (row[발생_col] / row["인구합계"] * 100000) if row["인구합계"] > 0 else 0.0,
                    axis=1
                )
            else:
                logger.warning(f"컬럼을 찾을 수 없습니다: {발생_col}")
                grouped_df[rate_col] = 0.0
        
        # 최종 컬럼 선택: 자치구, 관서명, 각 범죄별 범죄율
        result_columns = ["자치구", "관서명"] + list(crime_rate_mapping.keys())
        result_df = grouped_df[result_columns].copy()
        
        # 범죄율을 소수점 2자리로 반올림
        for rate_col in crime_rate_mapping.keys():
            result_df[rate_col] = result_df[rate_col].round(2)
        
        logger.info(f"[load_crime] 범죄율 데이터 전처리 완료: {len(result_df)}행")
        logger.info(f"[load_crime] 최종 컬럼: {list(result_df.columns)}")
        
        return result_df

    def load_crime_arrest_rate(self, filename: str = "crime.csv") -> pd.DataFrame:
        """
        crime.csv 파일을 읽어서 관서명을 Naver API로 자치구를 찾고,
        같은 구 단위로 합계를 내서 각 범죄별 검거율을 계산하여 DataFrame으로 반환
        
        Args:
            filename: 범죄 데이터 파일명 (기본값: crime.csv)
            
        Returns:
            자치구, 관서명(쉼표 구분), 각 범죄별 검거율을 포함한 DataFrame
            컬럼: 자치구, 관서명, 살인검거율, 강도검거율, 강간검거율, 절도검거율, 폭력검거율
        """
        crime_path = self.data_dir / filename
        
        # CSV 파일 읽기
        df = pd.read_csv(crime_path)
        
        logger.info(f"[load_crime_arrest_rate] 범죄 데이터 로드: {len(df)}행, 컬럼: {list(df.columns)}")
        
        # 관서명 컬럼 확인
        if "관서명" not in df.columns:
            raise ValueError(f"'관서명' 컬럼을 찾을 수 없습니다. 컬럼: {list(df.columns)}")
        
        # 각 관서명에 대해 Naver API로 자치구 조회
        districts = []
        
        for station_name in df["관서명"]:
            logger.info(f"[load_crime_arrest_rate] 관서명 조회 시작: {station_name}")
            try:
                station_info = self.naver_client.search_police_station(station_name)
                logger.info(f"[load_crime_arrest_rate] Naver API 응답 받음: {station_name}, 결과: {station_info is not None}")
            except Exception as e:
                logger.error(f"[load_crime_arrest_rate] Naver API 호출 중 오류 발생: {station_name}, 오류: {str(e)}", exc_info=True)
                station_info = None
            
            if station_info:
                district = station_info.get("district", "")
                logger.info(f"[load_crime_arrest_rate] 관서: {station_name} | 자치구: {district}")
                districts.append(district)
            else:
                logger.warning(f"관서 정보를 찾을 수 없습니다: {station_name}")
                districts.append(None)
        
        # 자치구 컬럼 추가
        df["자치구"] = districts
        
        # 숫자 컬럼 정리 (쉼표 제거 및 숫자 변환)
        numeric_columns = df.select_dtypes(include=["object"]).columns
        for col in numeric_columns:
            if col not in ["관서명", "자치구"]:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "")
                    .str.replace('"', "")
                    .str.replace("-", "0")
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 결측치 처리 (관서명, 자치구는 제외)
        numeric_cols_for_fillna = [c for c in df.columns if c not in ["관서명", "자치구"]]
        df[numeric_cols_for_fillna] = df[numeric_cols_for_fillna].fillna(0)
        
        # 자치구별로 그룹화하여 합계 계산
        # 발생 컬럼과 검거 컬럼 구분
        발생_컬럼들 = ["살인 발생", "강도 발생", "강간 발생", "절도 발생", "폭력 발생"]
        검거_컬럼들 = ["살인 검거", "강도 검거", "강간 검거", "절도 검거", "폭력 검거"]
        
        # 자치구별로 집계 (발생과 검거는 합계로)
        agg_dict = {}
        for col in 발생_컬럼들 + 검거_컬럼들:
            if col in df.columns:
                agg_dict[col] = "sum"
        
        # 관서명은 쉼표로 구분된 문자열로
        def join_stations(x):
            unique_stations = x.astype(str).unique()
            return ", ".join([s for s in unique_stations if s and s != 'nan'])
        agg_dict["관서명"] = join_stations
        
        # 자치구가 None인 행 제외
        df_with_district = df[df["자치구"].notna()].copy()
        
        # 자치구별로 집계
        grouped_df = df_with_district.groupby("자치구").agg(agg_dict).reset_index()
        
        # 각 범죄별 검거율 계산
        arrest_rate_mapping = {
            "살인검거율": ("살인 발생", "살인 검거"),
            "강도검거율": ("강도 발생", "강도 검거"),
            "강간검거율": ("강간 발생", "강간 검거"),
            "절도검거율": ("절도 발생", "절도 검거"),
            "폭력검거율": ("폭력 발생", "폭력 검거"),
        }
        
        for rate_col, (발생_col, 검거_col) in arrest_rate_mapping.items():
            if 발생_col in grouped_df.columns and 검거_col in grouped_df.columns:
                # 검거율 = (검거 / 발생) * 100
                # 발생이 0이면 검거율도 0으로 처리
                grouped_df[rate_col] = grouped_df.apply(
                    lambda row: (row[검거_col] / row[발생_col] * 100) if row[발생_col] > 0 else 0.0,
                    axis=1
                )
            else:
                logger.warning(f"컬럼을 찾을 수 없습니다: {발생_col} 또는 {검거_col}")
                grouped_df[rate_col] = 0.0
        
        # 최종 컬럼 선택: 자치구, 관서명, 각 범죄별 검거율
        result_columns = ["자치구", "관서명"] + list(arrest_rate_mapping.keys())
        result_df = grouped_df[result_columns].copy()
        
        # 검거율을 소수점 2자리로 반올림
        for rate_col in arrest_rate_mapping.keys():
            result_df[rate_col] = result_df[rate_col].round(2)
        
        logger.info(f"[load_crime_arrest_rate] 검거율 데이터 전처리 완료: {len(result_df)}행")
        logger.info(f"[load_crime_arrest_rate] 최종 컬럼: {list(result_df.columns)}")
        
        return result_df

    def merge_all(
        self,
        pop_filename: str = "pop.csv",
        cctv_filename: str = "cctv.csv",
        crime_filename: Optional[str] = "crime.csv",
        how: str = "inner",
    ) -> pd.DataFrame:
        """
        pop, cctv, crime DataFrame을 '자치구' 기준으로 병합
        
        Args:
            pop_filename: 인구 데이터 파일명 (기본값: pop.csv)
            cctv_filename: CCTV 데이터 파일명 (기본값: cctv.csv)
            crime_filename: 범죄 데이터 파일명 (None이면 제외, 기본값: crime.csv)
            how: 병합 방식 (inner/left/right/outer)
            
        Returns:
            자치구 기준으로 병합된 DataFrame
        """
        # 각 DataFrame 로드
        pop_df = self.load_pop(pop_filename)
        cctv_df = self.load_cctv(cctv_filename)
        
        # 모든 DataFrame의 컬럼을 '자치구'로 통일
        # pop_df: '구' → '자치구'
        pop_df = pop_df.rename(columns={"구": "자치구"})
        
        # cctv_df: '기관명' → '자치구'
        cctv_df = cctv_df.rename(columns={"기관명": "자치구"})
        
        # pop + cctv 병합 (자치구 기준)
        merged = pd.merge(pop_df, cctv_df, on="자치구", how=how)
        logger.info(f"인구 + CCTV 데이터 병합 완료: {len(merged)}행, {len(merged.columns)}열")
        
        # crime 데이터가 있으면 추가 병합
        if crime_filename and crime_filename.lower() not in ['none', 'null', '']:
            # pop_filename을 load_crime에 전달 (범죄율 계산에 필요)
            crime_df = self.load_crime(crime_filename, pop_filename)
            
            # crime_df는 이미 '자치구' 컬럼을 가지고 있음
            # 같은 자치구에 여러 관서가 있을 수 있으므로 자치구별로 집계
            # 범죄율 컬럼은 평균으로 집계 (같은 자치구에 여러 관서가 있을 경우)
            agg_dict = {}
            # 범죄율 컬럼들
            crime_rate_cols = ["살인범죄율", "강도범죄율", "강간범죄율", "절도범죄율", "폭력범죄율"]
            for col in crime_rate_cols:
                if col in crime_df.columns:
                    agg_dict[col] = "mean"  # 평균으로 집계
            
            # 관서명은 쉼표로 구분된 문자열로
            def join_stations(x):
                unique_stations = x.astype(str).unique()
                return ", ".join([s for s in unique_stations if s and s != 'nan'])
            agg_dict["관서명"] = join_stations
            
            # 집계할 컬럼이 있으면 집계 수행
            if agg_dict:
                crime_df = crime_df.groupby("자치구").agg(agg_dict).reset_index()
            
            # 병합 (자치구 기준)
            merged = pd.merge(merged, crime_df, on="자치구", how=how)
            logger.info(f"전체 데이터 병합 완료: {len(merged)}행, {len(merged.columns)}열")
        
        return merged

    def create_heatmap(
        self,
        pop_filename: str = "pop.csv",
        cctv_filename: str = "cctv.csv",
        crime_filename: Optional[str] = "crime.csv",
        how: str = "inner",
        figsize: tuple = (12, 8),
        cmap: str = "YlOrRd",
        annot: bool = True,
        fmt: str = ".1f",
    ) -> str:
        """
        병합된 DataFrame의 범죄율 데이터를 히트맵으로 시각화하여 base64 인코딩된 이미지로 반환
        
        Args:
            pop_filename: 인구 데이터 파일명 (기본값: pop.csv)
            cctv_filename: CCTV 데이터 파일명 (기본값: cctv.csv)
            crime_filename: 범죄 데이터 파일명 (None이면 제외, 기본값: crime.csv)
            how: 병합 방식 (inner/left/right/outer)
            figsize: 그래프 크기 (기본값: (12, 8))
            cmap: 색상 맵 (기본값: "YlOrRd")
            annot: 셀에 값 표시 여부 (기본값: True)
            fmt: 값 표시 형식 (기본값: ".1f")
            
        Returns:
            base64 인코딩된 이미지 문자열
        """
        # 데이터 병합
        merged_df = self.merge_all(
            pop_filename=pop_filename,
            cctv_filename=cctv_filename,
            crime_filename=crime_filename,
            how=how,
        )
        
        # 범죄율 컬럼만 선택
        crime_rate_cols = ["살인범죄율", "강도범죄율", "강간범죄율", "절도범죄율", "폭력범죄율"]
        
        # 존재하는 범죄율 컬럼만 필터링
        available_cols = [col for col in crime_rate_cols if col in merged_df.columns]
        
        if not available_cols:
            raise ValueError("범죄율 컬럼을 찾을 수 없습니다. crime 데이터가 포함되어 있는지 확인하세요.")
        
        # 자치구를 인덱스로, 범죄율 컬럼만 선택
        heatmap_data = merged_df.set_index("자치구")[available_cols]
        
        # 데이터 정규화 (Min-Max Scaling: 0~1 사이로 변환)
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        
        # NaN 처리 (분모가 0인 경우 등)
        heatmap_data = heatmap_data.fillna(0.0)
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 히트맵 생성
        sns.heatmap(
            heatmap_data,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': '범죄율 (정규화: 0~1)'},
            ax=ax,
            vmin=0.0,
            vmax=1.0,  # 0~1 범위 지정
        )
        
        # 제목 및 레이블 설정
        ax.set_title('서울시 자치구별 범죄율 히트맵', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('범죄 유형', fontsize=12)
        ax.set_ylabel('자치구', fontsize=12)
        
        # x축 레이블 회전
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close(fig)
        
        logger.info(f"히트맵 생성 완료: {len(heatmap_data)}행, {len(available_cols)}열")
        
        return img_base64

    def create_arrest_rate_heatmap(
        self,
        filename: str = "crime.csv",
        figsize: tuple = (12, 8),
        cmap: str = "RdYlGn",
        annot: bool = True,
        fmt: str = ".1f",
    ) -> str:
        """
        검거율 데이터를 히트맵으로 시각화하여 base64 인코딩된 이미지로 반환
        
        Args:
            filename: 범죄 데이터 파일명 (기본값: crime.csv)
            figsize: 그래프 크기 (기본값: (12, 8))
            cmap: 색상 맵 (기본값: "RdYlGn" - 빨강(낮음), 노랑(중간), 초록(높음))
            annot: 셀에 값 표시 여부 (기본값: True)
            fmt: 값 표시 형식 (기본값: ".1f")
            
        Returns:
            base64 인코딩된 이미지 문자열
        """
        # 검거율 데이터 로드
        arrest_rate_df = self.load_crime_arrest_rate(filename)
        
        # 검거율 컬럼만 선택
        arrest_rate_cols = ["살인검거율", "강도검거율", "강간검거율", "절도검거율", "폭력검거율"]
        
        # 존재하는 검거율 컬럼만 필터링
        available_cols = [col for col in arrest_rate_cols if col in arrest_rate_df.columns]
        
        if not available_cols:
            raise ValueError("검거율 컬럼을 찾을 수 없습니다. crime 데이터를 확인하세요.")
        
        # 자치구를 인덱스로, 검거율 컬럼만 선택
        heatmap_data = arrest_rate_df.set_index("자치구")[available_cols]
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 히트맵 생성
        sns.heatmap(
            heatmap_data,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': '검거율 (%)'},
            ax=ax,
            vmin=0,
            vmax=100,  # 검거율은 0~100% 범위
        )
        
        # 제목 및 레이블 설정
        ax.set_title('서울시 자치구별 범죄 검거율 히트맵', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('범죄 유형', fontsize=12)
        ax.set_ylabel('자치구', fontsize=12)
        
        # x축 레이블 회전
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close(fig)
        
        logger.info(f"검거율 히트맵 생성 완료: {len(heatmap_data)}행, {len(available_cols)}열")
        
        return img_base64

    def load_geo_data(self) -> Dict:
        """
        서울시 자치구 GeoJSON 데이터를 로드합니다.
        
        Returns:
            GeoJSON 딕셔너리
        """
        geo_path = self.data_dir / "kr-state.json"
        
        if not geo_path.exists():
            raise FileNotFoundError(f"GeoJSON 파일을 찾을 수 없습니다: {geo_path}")
        
        with open(geo_path, 'r', encoding='utf-8') as f:
            geo_data = json.load(f)
        
        logger.info(f"GeoJSON 데이터 로드 완료: {len(geo_data.get('features', []))}개 자치구")
        return geo_data

    def create_crime_map(
        self,
        crime_filename: str = "crime.csv",
        pop_filename: str = "pop.csv",
        map_location: list = [37.5665, 126.9780],
        zoom_start: int = 11,
        crime_rate_fill_color: str = "YlOrRd",
        arrest_rate_fill_color: str = "RdYlGn",
        fill_opacity: float = 0.7,
        line_opacity: float = 0.2,
    ) -> folium.Map:
        """
        서울시 자치구별 범죄율과 검거율을 지도로 시각화합니다.
        범죄율과 검거율은 서로 다른 레이어로 표시됩니다.
        
        Args:
            crime_filename: 범죄 데이터 파일명 (기본값: crime.csv)
            pop_filename: 인구 데이터 파일명 (기본값: pop.csv)
            map_location: 지도 중심 좌표 [위도, 경도] (기본값: 서울시청)
            zoom_start: 지도 초기 줌 레벨 (기본값: 11)
            crime_rate_fill_color: 범죄율 색상 팔레트 (기본값: "YlOrRd")
            arrest_rate_fill_color: 검거율 색상 팔레트 (기본값: "RdYlGn")
            fill_opacity: 채우기 투명도 (기본값: 0.7)
            line_opacity: 선 투명도 (기본값: 0.2)
            
        Returns:
            Folium Map 객체
        """
        # GeoJSON 데이터 로드
        geo_data = self.load_geo_data()
        
        # 범죄율 데이터 로드
        crime_rate_df = self.load_crime(crime_filename, pop_filename)
        
        # 검거율 데이터 로드
        arrest_rate_df = self.load_crime_arrest_rate(crime_filename)
        
        # 자치구별 평균 범죄율 계산 (여러 범죄 유형의 평균)
        crime_rate_cols = ["살인범죄율", "강도범죄율", "강간범죄율", "절도범죄율", "폭력범죄율"]
        available_crime_cols = [col for col in crime_rate_cols if col in crime_rate_df.columns]
        if available_crime_cols:
            crime_rate_df["평균범죄율"] = crime_rate_df[available_crime_cols].mean(axis=1)
        else:
            raise ValueError("범죄율 컬럼을 찾을 수 없습니다.")
        
        # 자치구별 평균 검거율 계산
        arrest_rate_cols = ["살인검거율", "강도검거율", "강간검거율", "절도검거율", "폭력검거율"]
        available_arrest_cols = [col for col in arrest_rate_cols if col in arrest_rate_df.columns]
        if available_arrest_cols:
            arrest_rate_df["평균검거율"] = arrest_rate_df[available_arrest_cols].mean(axis=1)
        else:
            raise ValueError("검거율 컬럼을 찾을 수 없습니다.")
        
        # 범죄율 데이터 준비 (자치구별로 집계 - 같은 자치구에 여러 행이 있을 경우 평균)
        crime_rate_data = crime_rate_df.groupby("자치구")["평균범죄율"].mean().reset_index()
        crime_rate_data = crime_rate_data.rename(columns={"평균범죄율": "범죄율"})
        
        # 검거율 데이터 준비 (자치구별로 집계 - 같은 자치구에 여러 행이 있을 경우 평균)
        arrest_rate_data = arrest_rate_df.groupby("자치구")["평균검거율"].mean().reset_index()
        arrest_rate_data = arrest_rate_data.rename(columns={"평균검거율": "검거율"})
        
        # 지도 생성
        logger.info(f"지도 생성 중: location={map_location}, zoom_start={zoom_start}")
        m = folium.Map(location=map_location, zoom_start=zoom_start)
        
        # 범죄율 Choropleth 레이어 추가
        logger.info("범죄율 Choropleth 레이어 추가 중")
        crime_choropleth = folium.Choropleth(
            geo_data=geo_data,
            name="범죄율 (10만 명당)",
            data=crime_rate_data,
            columns=["자치구", "범죄율"],
            key_on="feature.id",
            fill_color=crime_rate_fill_color,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
            legend_name="범죄율 (10만 명당)",
        )
        crime_choropleth.add_to(m)
        
        # 검거율 Choropleth 레이어 추가
        logger.info("검거율 Choropleth 레이어 추가 중")
        arrest_choropleth = folium.Choropleth(
            geo_data=geo_data,
            name="검거율 (%)",
            data=arrest_rate_data,
            columns=["자치구", "검거율"],
            key_on="feature.id",
            fill_color=arrest_rate_fill_color,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
            legend_name="검거율 (%)",
        )
        arrest_choropleth.add_to(m)
        
        # 레이어 컨트롤 추가
        logger.info("레이어 컨트롤 추가 중")
        folium.LayerControl(collapsed=False).add_to(m)
        
        logger.info("범죄율/검거율 지도 생성 완료")
        
        return m
