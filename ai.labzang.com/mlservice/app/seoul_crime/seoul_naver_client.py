"""
Naver API 클라이언트 유틸리티
- 관서명으로 검색하여 주소 정보 획득
- 주소에서 자치구 추출
- 네이버 지도 API (Geocoding/Reverse Geocoding) 지원
"""
import os
import re
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
import logging

logger = logging.getLogger(__name__)


class SeoulNaverClient:
    """Naver Local Search API 및 네이버 지도 API를 활용한 클라이언트"""

    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            logger.error(
                "Naver API 인증 정보가 없습니다. "
                "NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET 환경 변수를 확인하세요."
            )
        else:
            # API 키 확인 (보안을 위해 일부만 표시)
            logger.info(
                f"Naver API 인증 정보 로드 완료: "
                f"Client ID={self.client_id[:10]}..., "
                f"Client Secret={'*' * len(self.client_secret) if self.client_secret else 'None'}"
            )
        
        # API 엔드포인트
        self.search_base_url = "https://openapi.naver.com/v1/search/local.json"
        self.geocoding_base_url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
        self.reverse_geocoding_base_url = "https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc"
        
        # 캐시 파일 경로
        current_file = Path(__file__).resolve()
        self.cache_dir = current_file.parent / "data"
        self.cache_file = self.cache_dir / "police_station_to_district.json"
        self.detail_cache_file = self.cache_dir / "police_station_detail.json"
        
        # 캐시 로드 (기존 district 캐시)
        self.cache: Dict[str, str] = self._load_cache()
        # 상세 정보 캐시 (주소, 위도, 경도 포함)
        self.detail_cache: Dict[str, Dict[str, Any]] = self._load_detail_cache()
        
        # API 호출 간격 (Rate Limit 대응: 초당 1회)
        self.last_api_call_time = 0
        self.min_call_interval = 1.0  # 1초
        
        # API 인증 실패 플래그 (401 에러 발생 시 이후 호출 건너뛰기)
        self.api_auth_failed = False

    def _load_cache(self) -> Dict[str, str]:
        """캐시 파일에서 매핑 정보 로드 (기존 district만)"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    logger.info(f"캐시 파일 로드 완료: {len(cache)}개 매핑")
                    return cache
            except Exception as e:
                logger.error(f"캐시 파일 로드 실패: {e}")
                return {}
        return {}
    
    def _load_detail_cache(self) -> Dict[str, Dict[str, Any]]:
        """상세 정보 캐시 파일에서 로드 (주소, 위도, 경도 포함)"""
        if self.detail_cache_file.exists():
            try:
                with open(self.detail_cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    logger.info(f"상세 캐시 파일 로드 완료: {len(cache)}개 매핑")
                    return cache
            except Exception as e:
                logger.error(f"상세 캐시 파일 로드 실패: {e}")
                return {}
        return {}
    
    def _save_detail_cache(self):
        """상세 정보를 캐시 파일에 저장"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.detail_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.detail_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"상세 캐시 파일 저장 완료: {len(self.detail_cache)}개 매핑")
        except Exception as e:
            logger.error(f"상세 캐시 파일 저장 실패: {e}")

    def _save_cache(self):
        """매핑 정보를 캐시 파일에 저장"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"캐시 파일 저장 완료: {len(self.cache)}개 매핑")
        except Exception as e:
            logger.error(f"캐시 파일 저장 실패: {e}")

    def _rate_limit_wait(self):
        """Rate Limit을 위한 대기"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call_time
        if elapsed < self.min_call_interval:
            wait_time = self.min_call_interval - elapsed
            time.sleep(wait_time)
        self.last_api_call_time = time.time()

    def _extract_district_from_address(self, address: str) -> Optional[str]:
        """
        주소 문자열에서 자치구 추출
        예: "서울특별시 중구 을지로..." → "중구"
        """
        if not address:
            return None
        
        # 서울시 자치구 목록 (25개)
        seoul_districts = [
            "종로구", "중구", "용산구", "성동구", "광진구",
            "동대문구", "중랑구", "성북구", "강북구", "도봉구",
            "노원구", "은평구", "서대문구", "마포구", "양천구",
            "강서구", "구로구", "금천구", "영등포구", "동작구",
            "관악구", "서초구", "강남구", "송파구", "강동구"
        ]
        
        # 주소에서 자치구 찾기
        for district in seoul_districts:
            if district in address:
                return district
        
        # 정규표현식으로 패턴 매칭 시도
        patterns = [
            r"서울[시특별시]*\s*([가-힣]+구)",
            r"([가-힣]+구)\s",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, address)
            if match:
                district = match.group(1)
                if district in seoul_districts:
                    return district
        
        return None

    def search_police_station(self, station_name: str) -> Optional[Dict[str, Any]]:
        """
        관서명으로 검색하여 자치구, 주소, 위도, 경도 반환
        
        Args:
            station_name: 관서명 (예: "중부서", "강남서")
            
        Returns:
            {district, address, latitude, longitude} 딕셔너리 또는 None
        """
        if not station_name:
            return None
        
        # 상세 캐시 확인 (주소, 위도, 경도 포함)
        if station_name in self.detail_cache:
            cached_data = self.detail_cache[station_name]
            logger.info(
                f"[Naver API] 캐시에서 발견: {station_name} | "
                f"주소: {cached_data.get('address', '')} | "
                f"위도: {cached_data.get('latitude')} | "
                f"경도: {cached_data.get('longitude')} | "
                f"자치구: {cached_data.get('district', '')}"
            )
            return cached_data
        
        # API 인증 실패 플래그 확인 (이전에 401 에러가 발생했다면 건너뛰기)
        if self.api_auth_failed:
            logger.info(f"API 인증 실패로 인해 건너뜀: {station_name}")
            return None
        
        # API 인증 정보 확인
        if not self.client_id or not self.client_secret:
            logger.error("Naver API 인증 정보가 없습니다.")
            return None
        
        # Rate Limit 대기
        self._rate_limit_wait()
        
        # 검색 쿼리 구성 (서울 + 관서명)
        query = f"서울 {station_name}"
        
        try:
            headers = {
                "X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret,
            }
            
            params = {
                "query": query,
                "display": 1,  # 최상위 1개 결과만
            }
            
            logger.info(f"Naver API 호출: {query}")
            response = requests.get(
                self.search_base_url, headers=headers, params=params, timeout=10
            )
            
            # 401 오류 처리
            if response.status_code == 401:
                error_detail = response.text
                logger.error(
                    f"Naver API 인증 실패 (401): "
                    f"Client ID가 '{self.client_id[:10]}...'로 시작합니다. "
                    f"API 키에 'Search' API 권한이 있는지 확인하세요. "
                    f"응답: {error_detail}"
                )
                # 이후 모든 API 호출 건너뛰기
                self.api_auth_failed = True
                logger.warning("Naver API 인증 실패 플래그 설정됨. 이후 모든 API 호출이 건너뛰어집니다.")
                return None
            
            response.raise_for_status()
            
            data = response.json()
            
            # 결과 파싱
            if "items" in data and len(data["items"]) > 0:
                item = data["items"][0]
                address = item.get("roadAddress") or item.get("address", "")
                
                if address:
                    district = self._extract_district_from_address(address)
                    if district:
                        # 위도, 경도 추출
                        mapy = item.get("mapy", "")  # 위도
                        mapx = item.get("mapx", "")  # 경도
                        
                        # 위도, 경도를 숫자로 변환 (문자열일 수 있음)
                        try:
                            latitude = float(mapy) / 10000000 if mapy else None
                            longitude = float(mapx) / 10000000 if mapx else None
                        except (ValueError, TypeError):
                            latitude = None
                            longitude = None
                        
                        result = {
                            "district": district,
                            "address": address,
                            "latitude": latitude,
                            "longitude": longitude
                        }
                        
                        # 상세 캐시에 저장
                        self.detail_cache[station_name] = result
                        self._save_detail_cache()
                        
                        # 기존 캐시에도 저장 (하위 호환성)
                        self.cache[station_name] = district
                        self._save_cache()
                        
                        logger.info(f"매핑 완료: {station_name} → {district} (주소: {address}, 위도: {latitude}, 경도: {longitude})")
                        return result
                    else:
                        logger.warning(
                            f"주소에서 자치구를 추출할 수 없습니다: {address}"
                        )
                else:
                    logger.warning(f"주소 정보가 없습니다: {item}")
            else:
                logger.warning(f"검색 결과가 없습니다: {query}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Naver API 호출 실패: {e}")
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}", exc_info=True)
        
        return None

    def batch_search(self, station_names: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        여러 관서명을 일괄 검색
        
        Args:
            station_names: 관서명 리스트
            
        Returns:
            {관서명: {district, address, latitude, longitude}} 딕셔너리
        """
        result = {}
        
        # 중복 제거
        unique_stations = list(set(station_names))
        
        logger.info(f"일괄 검색 시작: {len(unique_stations)}개 관서")
        
        for station in unique_stations:
            station_info = self.search_police_station(station)
            if station_info:
                result[station] = station_info
            else:
                logger.warning(f"매핑 실패: {station}")
        
        logger.info(f"일괄 검색 완료: {len(result)}/{len(unique_stations)}개 성공")
        return result

    def geocode(self, query: str) -> Optional[Dict[str, Any]]:
        """
        네이버 지도 Geocoding API: 주소를 좌표로 변환
        
        Args:
            query: 검색할 주소 (예: "서울특별시 강남구 테헤란로 152")
            
        Returns:
            {
                "address": 주소,
                "latitude": 위도,
                "longitude": 경도,
                "roadAddress": 도로명 주소,
                "jibunAddress": 지번 주소
            } 또는 None
        """
        if not query:
            return None
        
        # API 인증 정보 확인
        if not self.client_id or not self.client_secret:
            logger.error("Naver API 인증 정보가 없습니다.")
            return None
        
        # Rate Limit 대기
        self._rate_limit_wait()
        
        try:
            headers = {
                "X-NCP-APIGW-API-KEY-ID": self.client_id,
                "X-NCP-APIGW-API-KEY": self.client_secret,
            }
            
            params = {
                "query": query,
            }
            
            logger.info(f"네이버 지도 Geocoding API 호출: {query}")
            response = requests.get(
                self.geocoding_base_url, headers=headers, params=params, timeout=10
            )
            
            # 401 오류 처리
            if response.status_code == 401:
                error_detail = response.text
                logger.error(
                    f"네이버 지도 API 인증 실패 (401): "
                    f"Client ID가 '{self.client_id[:10]}...'로 시작합니다. "
                    f"API 키에 'Geocoding' API 권한이 있는지 확인하세요. "
                    f"응답: {error_detail}"
                )
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # 결과 파싱
            if data.get("status") == "OK" and data.get("addresses"):
                address_info = data["addresses"][0]  # 첫 번째 결과 사용
                
                result = {
                    "address": address_info.get("roadAddress") or address_info.get("jibunAddress", ""),
                    "roadAddress": address_info.get("roadAddress", ""),
                    "jibunAddress": address_info.get("jibunAddress", ""),
                    "latitude": float(address_info.get("y", 0)) if address_info.get("y") else None,
                    "longitude": float(address_info.get("x", 0)) if address_info.get("x") else None,
                }
                
                logger.info(f"Geocoding 완료: {query} → 위도: {result['latitude']}, 경도: {result['longitude']}")
                return result
            else:
                logger.warning(f"Geocoding 결과가 없습니다: {query}, 상태: {data.get('status')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"네이버 지도 Geocoding API 호출 실패: {e}")
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}", exc_info=True)
        
        return None

    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        네이버 지도 Reverse Geocoding API: 좌표를 주소로 변환
        
        Args:
            latitude: 위도
            longitude: 경도
            
        Returns:
            {
                "address": 주소,
                "roadAddress": 도로명 주소,
                "jibunAddress": 지번 주소,
                "sido": 시도,
                "sigungu": 시군구,
                "dong": 동,
                "latitude": 위도,
                "longitude": 경도
            } 또는 None
        """
        if latitude is None or longitude is None:
            return None
        
        # API 인증 정보 확인
        if not self.client_id or not self.client_secret:
            logger.error("Naver API 인증 정보가 없습니다.")
            return None
        
        # Rate Limit 대기
        self._rate_limit_wait()
        
        try:
            headers = {
                "X-NCP-APIGW-API-KEY-ID": self.client_id,
                "X-NCP-APIGW-API-KEY": self.client_secret,
            }
            
            # 좌표를 문자열로 변환 (네이버 API 형식)
            coords = f"{longitude},{latitude}"
            
            params = {
                "request": "coordsToaddr",
                "coords": coords,
                "output": "json",
            }
            
            logger.info(f"네이버 지도 Reverse Geocoding API 호출: 위도={latitude}, 경도={longitude}")
            response = requests.get(
                self.reverse_geocoding_base_url, headers=headers, params=params, timeout=10
            )
            
            # 401 오류 처리
            if response.status_code == 401:
                error_detail = response.text
                logger.error(
                    f"네이버 지도 API 인증 실패 (401): "
                    f"Client ID가 '{self.client_id[:10]}...'로 시작합니다. "
                    f"API 키에 'Reverse Geocoding' API 권한이 있는지 확인하세요. "
                    f"응답: {error_detail}"
                )
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # 결과 파싱
            if data.get("status") == "OK" and data.get("results"):
                result_info = data["results"][0]  # 첫 번째 결과 사용
                region = result_info.get("region", {})
                land = result_info.get("land", {})
                
                # 주소 구성
                sido = region.get("area1", {}).get("name", "")
                sigungu = region.get("area2", {}).get("name", "")
                dong = region.get("area3", {}).get("name", "")
                
                road_address = land.get("name", "")
                jibun_address = land.get("number1", "")
                
                full_address = f"{sido} {sigungu} {dong} {road_address}".strip()
                
                result = {
                    "address": full_address,
                    "roadAddress": f"{sido} {sigungu} {road_address}".strip() if road_address else "",
                    "jibunAddress": f"{sido} {sigungu} {dong} {jibun_address}".strip() if jibun_address else "",
                    "sido": sido,
                    "sigungu": sigungu,
                    "dong": dong,
                    "latitude": latitude,
                    "longitude": longitude,
                }
                
                logger.info(f"Reverse Geocoding 완료: 위도={latitude}, 경도={longitude} → {full_address}")
                return result
            else:
                logger.warning(f"Reverse Geocoding 결과가 없습니다: 위도={latitude}, 경도={longitude}, 상태: {data.get('status')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"네이버 지도 Reverse Geocoding API 호출 실패: {e}")
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}", exc_info=True)
        
        return None

    def batch_geocode(self, addresses: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        여러 주소를 일괄 Geocoding
        
        Args:
            addresses: 주소 리스트
            
        Returns:
            {주소: {address, latitude, longitude, ...}} 딕셔너리
        """
        result = {}
        
        # 중복 제거
        unique_addresses = list(set(addresses))
        
        logger.info(f"일괄 Geocoding 시작: {len(unique_addresses)}개 주소")
        
        for address in unique_addresses:
            geocode_result = self.geocode(address)
            if geocode_result:
                result[address] = geocode_result
            else:
                logger.warning(f"Geocoding 실패: {address}")
        
        logger.info(f"일괄 Geocoding 완료: {len(result)}/{len(unique_addresses)}개 성공")
        return result

