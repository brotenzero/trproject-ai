"""
NLP(자연어 처리) 관련 라우터
"""
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import sys
import base64
import io

# 공통 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.nlp.nlp_service import NLPService
from common.utils import create_response, create_error_response
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nlp", tags=["nlp"])

# 서비스 인스턴스 생성 (싱글톤 패턴)
_service_instance: Optional[NLPService] = None


def get_service() -> NLPService:
    """NLPService 싱글톤 인스턴스 반환"""
    global _service_instance
    if _service_instance is None:
        _service_instance = NLPService()
    return _service_instance


# Pydantic 모델 정의
class TextInput(BaseModel):
    """텍스트 입력 모델"""
    text: str = Field(..., description="분석할 텍스트", min_length=1)
    name: Optional[str] = Field("Document", description="문서 이름")
    tokenize_method: Optional[str] = Field("regexp", description="토큰화 방법: word, sentence, regexp")


class TokenizeInput(BaseModel):
    """토큰화 입력 모델"""
    text: str = Field(..., description="토큰화할 텍스트", min_length=1)
    method: Optional[str] = Field("word", description="토큰화 방법: word, sentence, regexp")


class StemInput(BaseModel):
    """어간 추출 입력 모델"""
    words: List[str] = Field(..., description="어간 추출할 단어 리스트", min_items=1)
    method: Optional[str] = Field("porter", description="어간 추출 방법: porter, lancaster")


class LemmatizeInput(BaseModel):
    """원형 복원 입력 모델"""
    words: List[str] = Field(..., description="원형 복원할 단어 리스트", min_items=1)
    pos: Optional[str] = Field(None, description="품사: v(동사), n(명사), a(형용사), r(부사)")


class POSTagInput(BaseModel):
    """품사 태깅 입력 모델"""
    text: str = Field(..., description="품사 태깅할 텍스트", min_length=1)
    filter_pos: Optional[str] = Field(None, description="필터링할 품사 태그 (예: NN, NNP, VB)")


@router.get("/")
async def nlp_root():
    """NLP 서비스 루트"""
    return create_response(
        data={"service": "mlservice", "module": "nlp", "status": "running"},
        message="NLP Service is running"
    )


@router.get("/health")
async def health_check():
    """헬스 체크"""
    try:
        service = get_service()
        files = service.corpus_manager.get_available_files()
        return create_response(
            data={"status": "healthy", "service": "nlp", "available_corpus_files": len(files)},
            message="NLP service is healthy"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")


@router.get("/corpus/files")
async def get_corpus_files():
    """
    사용 가능한 말뭉치 파일 목록 조회
    
    NLTK Gutenberg 말뭉치에서 사용 가능한 파일들을 반환합니다.
    """
    try:
        service = get_service()
        files = service.corpus_manager.get_available_files()
        
        return create_response(
            data={
                "count": len(files),
                "files": files
            },
            message="말뭉치 파일 목록 조회 완료"
        )
    except Exception as e:
        logger.error(f"말뭉치 파일 목록 조회 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"말뭉치 파일 목록 조회 중 오류: {str(e)}")


@router.get("/corpus/preview")
async def preview_corpus(
    file_id: str = Query("austen-emma.txt", description="말뭉치 파일 ID"),
    length: int = Query(1000, description="미리보기 길이", ge=100, le=10000)
):
    """
    말뭉치 파일 미리보기
    
    지정된 말뭉치 파일의 일부분을 반환합니다.
    """
    try:
        service = get_service()
        preview = service.corpus_manager.preview_corpus(file_id, length)
        
        return create_response(
            data={
                "file_id": file_id,
                "preview_length": len(preview),
                "preview": preview
            },
            message="말뭉치 미리보기 조회 완료"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except Exception as e:
        logger.error(f"말뭉치 미리보기 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"말뭉치 미리보기 중 오류: {str(e)}")


@router.post("/tokenize")
async def tokenize_text(input_data: TokenizeInput):
    """
    텍스트 토큰화
    
    입력 텍스트를 선택한 방법으로 토큰화합니다.
    - word: 단어 단위 토큰화
    - sentence: 문장 단위 토큰화
    - regexp: 정규표현식 토큰화
    """
    try:
        service = get_service()
        
        if input_data.method == "word":
            tokens = service.tokenizer.tokenize_words(input_data.text)
        elif input_data.method == "sentence":
            tokens = service.tokenizer.tokenize_sentences(input_data.text)
        elif input_data.method == "regexp":
            tokens = service.tokenizer.tokenize_regexp(input_data.text)
        else:
            raise ValueError(f"지원하지 않는 토큰화 방법: {input_data.method}")
        
        return create_response(
            data={
                "method": input_data.method,
                "token_count": len(tokens),
                "tokens": tokens
            },
            message="토큰화 완료"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"토큰화 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"토큰화 중 오류: {str(e)}")


@router.post("/stem")
async def stem_words(input_data: StemInput):
    """
    어간 추출 (Stemming)
    
    단어의 접미사나 어미를 제거하여 기본형을 찾습니다.
    - porter: Porter Stemmer (보편적)
    - lancaster: Lancaster Stemmer (더 공격적)
    """
    try:
        service = get_service()
        
        if input_data.method == "porter":
            stems = service.morphology.stem_porter(input_data.words)
        elif input_data.method == "lancaster":
            stems = service.morphology.stem_lancaster(input_data.words)
        else:
            raise ValueError(f"지원하지 않는 어간 추출 방법: {input_data.method}")
        
        # 원본과 결과를 매핑
        result_pairs = [{"original": orig, "stem": stem} 
                       for orig, stem in zip(input_data.words, stems)]
        
        return create_response(
            data={
                "method": input_data.method,
                "count": len(stems),
                "results": result_pairs
            },
            message="어간 추출 완료"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"어간 추출 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"어간 추출 중 오류: {str(e)}")


@router.post("/lemmatize")
async def lemmatize_words(input_data: LemmatizeInput):
    """
    원형 복원 (Lemmatizing)
    
    단어를 사전형으로 통일합니다. 품사를 지정하면 더 정확한 결과를 얻을 수 있습니다.
    - pos: v(동사), n(명사), a(형용사), r(부사)
    """
    try:
        service = get_service()
        lemmas = service.morphology.lemmatize(input_data.words, input_data.pos)
        
        # 원본과 결과를 매핑
        result_pairs = [{"original": orig, "lemma": lemma} 
                       for orig, lemma in zip(input_data.words, lemmas)]
        
        return create_response(
            data={
                "pos": input_data.pos or "auto",
                "count": len(lemmas),
                "results": result_pairs
            },
            message="원형 복원 완료"
        )
    except Exception as e:
        logger.error(f"원형 복원 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"원형 복원 중 오류: {str(e)}")


@router.post("/pos-tag")
async def pos_tag_text(input_data: POSTagInput):
    """
    품사 태깅 (POS Tagging)
    
    텍스트를 토큰화하고 각 토큰에 품사를 부착합니다.
    - filter_pos를 지정하면 해당 품사만 필터링합니다.
    
    주요 품사 태그:
    - NN: 명사(단수)
    - NNP: 고유명사(단수)
    - VB: 동사
    - JJ: 형용사
    - RB: 부사
    """
    try:
        service = get_service()
        
        # 토큰화
        tokens = service.tokenizer.tokenize_words(input_data.text)
        
        # 품사 태깅
        tagged = service.pos_tagger.tag(tokens)
        
        # 필터링 (선택적)
        if input_data.filter_pos:
            filtered_tokens = service.pos_tagger.filter_by_pos(tagged, input_data.filter_pos)
            return create_response(
                data={
                    "total_tokens": len(tokens),
                    "tagged_tokens": tagged,
                    "filter_pos": input_data.filter_pos,
                    "filtered_count": len(filtered_tokens),
                    "filtered_tokens": filtered_tokens
                },
                message="품사 태깅 및 필터링 완료"
            )
        
        return create_response(
            data={
                "total_tokens": len(tokens),
                "tagged_tokens": tagged
            },
            message="품사 태깅 완료"
        )
    except Exception as e:
        logger.error(f"품사 태깅 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"품사 태깅 중 오류: {str(e)}")


@router.post("/analyze")
async def analyze_text(input_data: TextInput):
    """
    텍스트 종합 분석
    
    입력 텍스트에 대해 다음 분석을 수행합니다:
    - 토큰화
    - 빈도 분석
    - 가장 빈번한 단어 추출
    - 통계 정보
    """
    try:
        service = get_service()
        
        # 텍스트 분석기 생성
        analyzer = service.create_analyzer(
            text=input_data.text,
            name=input_data.name,
            tokenize_method=input_data.tokenize_method
        )
        
        # 분석 수행
        freq_dist = analyzer.get_freq_dist()
        most_common = analyzer.most_common(20)
        
        return create_response(
            data={
                "document_name": input_data.name,
                "total_tokens": len(analyzer.tokens),
                "unique_tokens": len(freq_dist),
                "most_common_words": [
                    {"word": word, "count": count}
                    for word, count in most_common
                ],
                "lexical_diversity": round(len(freq_dist) / len(analyzer.tokens), 4) if analyzer.tokens else 0
            },
            message="텍스트 분석 완료"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"텍스트 분석 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"텍스트 분석 중 오류: {str(e)}")


@router.get("/corpus/analyze")
async def analyze_corpus(
    file_id: str = Query("austen-emma.txt", description="분석할 말뭉치 파일 ID"),
    top_n: int = Query(20, description="상위 N개 단어 추출", ge=5, le=100)
):
    """
    말뭉치 전체 분석
    
    지정된 말뭉치 파일에 대해 종합 분석을 수행합니다.
    - 토큰 통계
    - 빈도 분석
    - 고유명사 추출
    """
    try:
        service = get_service()
        
        # 말뭉치 분석
        result = service.analyze_corpus(file_id)
        
        # 고유명사 추출
        stopwords = ["Mr.", "Mrs.", "Miss", "Mr", "Mrs", "Dear"]
        proper_nouns_fd = result['analyzer'].filter_proper_nouns(stopwords)
        proper_nouns_top = proper_nouns_fd.most_common(top_n)
        
        return create_response(
            data={
                "file_id": result['file_id'],
                "total_tokens": result['total_tokens'],
                "unique_tokens": result['unique_tokens'],
                "lexical_diversity": round(result['unique_tokens'] / result['total_tokens'], 4),
                "most_common_words": [
                    {"word": word, "count": count}
                    for word, count in result['most_common_words']
                ],
                "proper_nouns_top": [
                    {"word": word, "count": count}
                    for word, count in proper_nouns_top
                ]
            },
            message="말뭉치 분석 완료"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except Exception as e:
        logger.error(f"말뭉치 분석 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"말뭉치 분석 중 오류: {str(e)}")


@router.get("/wordcloud", response_class=HTMLResponse)
async def generate_wordcloud(
    file_id: str = Query("austen-emma.txt", description="분석할 말뭉치 파일 ID"),
    width: int = Query(1000, description="워드클라우드 너비", ge=400, le=2000),
    height: int = Query(600, description="워드클라우드 높이", ge=300, le=1500),
    background_color: str = Query("white", description="배경색"),
    filter_type: str = Query("proper_nouns", description="필터 타입: all, proper_nouns")
):
    """
    워드클라우드 생성
    
    말뭉치의 단어 빈도를 시각화한 워드클라우드를 HTML로 반환합니다.
    - filter_type='all': 모든 단어
    - filter_type='proper_nouns': 고유명사만
    """
    try:
        service = get_service()
        
        # 말뭉치 분석
        result = service.analyze_corpus(file_id)
        
        # 필터링
        if filter_type == "proper_nouns":
            stopwords = ["Mr.", "Mrs.", "Miss", "Mr", "Mrs", "Dear"]
            freq_dist = result['analyzer'].filter_proper_nouns(stopwords)
            title = f"{file_id} - 고유명사 워드클라우드"
        else:
            freq_dist = result['freq_dist']
            title = f"{file_id} - 전체 단어 워드클라우드"
        
        # 워드클라우드 생성
        import matplotlib
        matplotlib.use('Agg')  # GUI 없이 사용
        import matplotlib.pyplot as plt
        
        wc = service.visualizer.generate_wordcloud(
            freq_dist,
            width=width,
            height=height,
            background_color=background_color,
            show=False
        )
        
        # 이미지를 base64로 변환
        buf = io.BytesIO()
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # HTML 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 10px;
                    font-size: 2em;
                }}
                .subtitle {{
                    text-align: center;
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 1.1em;
                }}
                .image-container {{
                    text-align: center;
                    margin: 30px 0;
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 8px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }}
                .info {{
                    margin-top: 30px;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .info h2 {{
                    color: #667eea;
                    margin-top: 0;
                    font-size: 1.3em;
                }}
                .info ul {{
                    color: #555;
                    line-height: 1.8;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}
                .stat-card {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }}
                .stat-label {{
                    color: #888;
                    font-size: 0.9em;
                    margin-bottom: 5px;
                }}
                .stat-value {{
                    color: #333;
                    font-size: 1.5em;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 {title}</h1>
                <p class="subtitle">NLTK 자연어 처리 워드클라우드 시각화</p>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-label">총 토큰 수</div>
                        <div class="stat-value">{result['total_tokens']:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">고유 토큰 수</div>
                        <div class="stat-value">{result['unique_tokens']:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">어휘 다양성</div>
                        <div class="stat-value">{result['unique_tokens'] / result['total_tokens']:.3f}</div>
                    </div>
                </div>
                
                <div class="image-container">
                    <img src="data:image/png;base64,{img_base64}" alt="워드클라우드" />
                </div>
                
                <div class="info">
                    <h2>ℹ️ 워드클라우드 정보</h2>
                    <ul>
                        <li><strong>파일:</strong> {file_id}</li>
                        <li><strong>필터 타입:</strong> {"고유명사만" if filter_type == "proper_nouns" else "전체 단어"}</li>
                        <li><strong>크기:</strong> {width} × {height} px</li>
                        <li><strong>배경색:</strong> {background_color}</li>
                        <li><strong>설명:</strong> 글자 크기는 해당 단어의 출현 빈도에 비례합니다.</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"워드클라우드 생성 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"워드클라우드 생성 중 오류: {str(e)}")

