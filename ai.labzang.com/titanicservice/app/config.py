"""
Titanic Service 설정
"""
from common.config import BaseServiceConfig


class TitanicServiceConfig(BaseServiceConfig):
    """타이타닉 서비스 설정"""
    service_name: str = "titanicservice"
    service_version: str = "1.0.0"
    port: int = 9002
    
    class Config:
        env_file = ".env"
        case_sensitive = False

