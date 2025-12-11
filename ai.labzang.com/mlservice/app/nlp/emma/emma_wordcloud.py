"""
NLTK 자연어 처리 패키지 - OOP 버전

https://datascienceschool.net/view-notebook/118731eec74b4ad3bdd2f89bab077e1b/

NLTK(Natural Language Toolkit) 패키지는 
교육용으로 개발된 자연어 처리 및 문서 분석용 파이썬 패키지다. 
다양한 기능 및 예제를 가지고 있으며 실무 및 연구에서도 많이 사용된다.

주요 기능:
- 말뭉치(corpus) 관리
- 토큰 생성(tokenizing)
- 형태소 분석(morphological analysis)
- 품사 태깅(POS tagging)
- 텍스트 분석 및 시각화
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag, untag
from nltk import Text, FreqDist
from nltk.corpus import gutenberg
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import os


class CorpusManager:
    """
    말뭉치(corpus) 관리 클래스
    
    말뭉치는 자연어 분석 작업을 위해 만든 샘플 문서 집합을 말한다.
    NLTK 패키지의 corpus 서브패키지에서는 다양한 연구용 말뭉치를 제공한다.
    """
    
    def __init__(self):
        """NLTK 말뭉치 초기화"""
        self._download_required_data()
    
    @staticmethod
    def _download_required_data():
        """필요한 NLTK 데이터 다운로드"""
        try:
            nltk.download('book', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            # 다운로드 실패해도 서비스는 계속 실행
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"NLTK data download warning: {e}")
    
    def get_available_files(self) -> List[str]:
        """Gutenberg 말뭉치에서 사용 가능한 파일 목록 반환"""
        return gutenberg.fileids()
    
    def load_corpus(self, file_id: str) -> str:
        """
        지정된 말뭉치 파일 로드
        
        Args:
            file_id: 말뭉치 파일 ID (예: "austen-emma.txt")
            
        Returns:
            원문 텍스트 문자열
        """
        return gutenberg.raw(file_id)
    
    def preview_corpus(self, file_id: str, length: int = 1000) -> str:
        """
        말뭉치 미리보기
        
        Args:
            file_id: 말뭉치 파일 ID
            length: 미리보기 길이
            
        Returns:
            텍스트 일부분
        """
        raw_text = self.load_corpus(file_id)
        return raw_text[:length]


class TextTokenizer:
    """
    텍스트 토큰화 클래스
    
    자연어 문서를 분석하기 위해 긴 문자열을 작은 단위(토큰)로 나눈다.
    영문의 경우 문장, 단어 등을 토큰으로 사용하거나 정규 표현식을 사용할 수 있다.
    """
    
    def __init__(self, pattern: str = r"[\w]+"):
        """
        토큰화 객체 초기화
        
        Args:
            pattern: 정규표현식 패턴
        """
        self.regexp_tokenizer = RegexpTokenizer(pattern)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        문장 단위로 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            문장 리스트
        """
        return sent_tokenize(text)
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        단어 단위로 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            단어 리스트
        """
        return word_tokenize(text)
    
    def tokenize_regexp(self, text: str) -> List[str]:
        """
        정규표현식을 사용한 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 리스트
        """
        return self.regexp_tokenizer.tokenize(text)


class MorphologicalAnalyzer:
    """
    형태소 분석 클래스
    
    형태소는 일정한 의미가 있는 가장 작은 말의 단위이다.
    단어로부터 어근, 접두사, 접미사, 품사 등을 파악하여 형태소를 처리한다.
    
    주요 기능:
    - 어간 추출(stemming)
    - 원형 복원(lemmatizing)
    """
    
    def __init__(self):
        """형태소 분석기 초기화"""
        self.porter_stemmer = PorterStemmer()
        self.lancaster_stemmer = LancasterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def stem_porter(self, words: List[str]) -> List[str]:
        """
        Porter Stemmer를 사용한 어간 추출
        
        Args:
            words: 단어 리스트
            
        Returns:
            어간 추출된 단어 리스트
        """
        return [self.porter_stemmer.stem(w) for w in words]
    
    def stem_lancaster(self, words: List[str]) -> List[str]:
        """
        Lancaster Stemmer를 사용한 어간 추출
        
        Args:
            words: 단어 리스트
            
        Returns:
            어간 추출된 단어 리스트
        """
        return [self.lancaster_stemmer.stem(w) for w in words]
    
    def lemmatize(self, words: List[str], pos: Optional[str] = None) -> List[str]:
        """
        원형 복원 (같은 의미를 가지는 여러 단어를 사전형으로 통일)
        
        Args:
            words: 단어 리스트
            pos: 품사 (예: 'v' for verb, 'n' for noun)
            
        Returns:
            원형 복원된 단어 리스트
        """
        if pos:
            return [self.lemmatizer.lemmatize(w, pos=pos) for w in words]
        return [self.lemmatizer.lemmatize(w) for w in words]


class POSTagger:
    """
    품사 태깅(Part-Of-Speech Tagging) 클래스
    
    품사는 낱말을 문법적인 기능이나 형태, 뜻에 따라 구분한 것이다.
    NLTK에서는 펜 트리뱅크 태그세트(Penn Treebank Tagset)를 사용한다.
    
    주요 태그:
    - NNP: 단수 고유명사
    - VB: 동사
    - VBP: 동사 현재형
    - TO: to 전치사
    - NN: 명사(단수형 혹은 집합형)
    - DT: 관형사
    """
    
    @staticmethod
    def tag(tokens: List[str]) -> List[Tuple[str, str]]:
        """
        토큰에 품사 태그 부착
        
        Args:
            tokens: 토큰 리스트
            
        Returns:
            (토큰, 품사) 튜플 리스트
        """
        return pos_tag(tokens)
    
    @staticmethod
    def untag(tagged_tokens: List[Tuple[str, str]]) -> List[str]:
        """
        품사 태그 제거
        
        Args:
            tagged_tokens: (토큰, 품사) 튜플 리스트
            
        Returns:
            토큰 리스트
        """
        return untag(tagged_tokens)
    
    @staticmethod
    def filter_by_pos(tagged_tokens: List[Tuple[str, str]], pos_tag: str) -> List[str]:
        """
        특정 품사의 토큰만 필터링
        
        Args:
            tagged_tokens: (토큰, 품사) 튜플 리스트
            pos_tag: 필터링할 품사 태그 (예: "NN")
            
        Returns:
            필터링된 토큰 리스트
        """
        return [token for token, tag in tagged_tokens if tag == pos_tag]
    
    @staticmethod
    def combine_token_pos(tagged_tokens: List[Tuple[str, str]], separator: str = "/") -> List[str]:
        """
        토큰과 품사를 결합하여 새로운 토큰 생성
        
        Args:
            tagged_tokens: (토큰, 품사) 튜플 리스트
            separator: 구분자
            
        Returns:
            결합된 토큰 리스트
        """
        return [f"{token}{separator}{tag}" for token, tag in tagged_tokens]
    
    @staticmethod
    def show_tagset_help(tag: str = None):
        """
        품사 태그세트 도움말 출력
        
        Args:
            tag: 특정 태그 (None이면 전체 출력)
        """
        if tag:
            nltk.help.upenn_tagset(tag)
        else:
            nltk.help.upenn_tagset()


class TextAnalyzer:
    """
    텍스트 분석 클래스
    
    NLTK의 Text 클래스를 활용하여 문서 분석을 수행한다.
    빈도 분석, 연어 추출, 문맥 분석 등의 기능을 제공한다.
    """
    
    def __init__(self, tokens: List[str], name: str = "Document"):
        """
        텍스트 분석기 초기화
        
        Args:
            tokens: 토큰 리스트
            name: 문서 이름
        """
        self.text = Text(tokens, name=name)
        self.tokens = tokens
        self.name = name
    
    def get_freq_dist(self) -> FreqDist:
        """
        빈도 분포 객체 반환
        
        Returns:
            FreqDist 객체
        """
        return self.text.vocab()
    
    def most_common(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        가장 빈번한 단어 추출
        
        Args:
            n: 추출할 단어 수
            
        Returns:
            (단어, 빈도) 튜플 리스트
        """
        fd = self.get_freq_dist()
        return fd.most_common(n)
    
    def word_frequency(self, word: str) -> Tuple[int, float]:
        """
        특정 단어의 출현 횟수와 확률 반환
        
        Args:
            word: 검색할 단어
            
        Returns:
            (출현횟수, 확률) 튜플
        """
        fd = self.get_freq_dist()
        return fd[word], fd.freq(word)
    
    def concordance(self, word: str, lines: int = 25):
        """
        단어가 사용된 위치 표시
        
        Args:
            word: 검색할 단어
            lines: 표시할 줄 수
        """
        self.text.concordance(word, lines=lines)
    
    def similar_words(self, word: str, num: int = 20):
        """
        해당 단어와 비슷한 문맥에서 사용된 단어 찾기
        
        Args:
            word: 기준 단어
            num: 찾을 단어 수
        """
        self.text.similar(word, num)
    
    def find_collocations(self, num: int = 20):
        """
        연어(같이 붙어서 쓰이는 단어) 찾기
        
        Args:
            num: 찾을 연어 수
        """
        self.text.collocations(num)
    
    def filter_proper_nouns(self, stopwords: List[str] = None) -> FreqDist:
        """
        고유명사만 추출하여 빈도 분포 생성
        
        Args:
            stopwords: 제외할 단어 리스트
            
        Returns:
            고유명사 FreqDist 객체
        """
        if stopwords is None:
            stopwords = []
        
        tagged_tokens = pos_tag(self.tokens)
        proper_nouns = [token for token, tag in tagged_tokens 
                       if tag == "NNP" and token not in stopwords]
        return FreqDist(proper_nouns)


class TextVisualizer:
    """
    텍스트 시각화 클래스
    
    빈도 분포, 단어 사용 위치, 워드클라우드 등을 시각화한다.
    """
    
    @staticmethod
    def plot_frequency(text: Text, num_words: int = 20, show: bool = True):
        """
        단어 사용 빈도 그래프 그리기
        
        Args:
            text: NLTK Text 객체
            num_words: 표시할 단어 수
            show: 즉시 표시 여부
        """
        text.plot(num_words)
        if show:
            plt.show()
    
    @staticmethod
    def plot_dispersion(text: Text, words: List[str], show: bool = True):
        """
        단어가 사용된 위치 시각화
        
        Args:
            text: NLTK Text 객체
            words: 시각화할 단어 리스트
            show: 즉시 표시 여부
        """
        text.dispersion_plot(words)
        if show:
            plt.show()
    
    @staticmethod
    def generate_wordcloud(freq_dist: FreqDist, 
                          width: int = 1000, 
                          height: int = 600,
                          background_color: str = "white",
                          random_state: int = 0,
                          show: bool = True) -> WordCloud:
        """
        워드클라우드 생성
        
        Args:
            freq_dist: 빈도 분포 객체
            width: 이미지 너비
            height: 이미지 높이
            background_color: 배경색
            random_state: 랜덤 시드
            show: 즉시 표시 여부
            
        Returns:
            WordCloud 객체
        """
        wc = WordCloud(width=width, height=height, 
                      background_color=background_color, 
                      random_state=random_state)
        wc.generate_from_frequencies(freq_dist)
        
        if show:
            plt.figure(figsize=(width/100, height/100))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.show()
        
        return wc


class NLPService:
    """
    통합 NLP 서비스 클래스
    
    모든 NLP 기능을 통합하여 제공하는 메인 서비스 클래스
    """
    
    def __init__(self):
        """NLP 서비스 초기화"""
        self.corpus_manager = CorpusManager()
        self.tokenizer = TextTokenizer()
        self.morphology = MorphologicalAnalyzer()
        self.pos_tagger = POSTagger()
        self.visualizer = TextVisualizer()
    
    def create_analyzer(self, text: str, name: str = "Document", 
                       tokenize_method: str = "regexp") -> TextAnalyzer:
        """
        텍스트 분석기 생성
        
        Args:
            text: 분석할 텍스트
            name: 문서 이름
            tokenize_method: 토큰화 방법 ("word", "sentence", "regexp")
            
        Returns:
            TextAnalyzer 객체
        """
        if tokenize_method == "word":
            tokens = self.tokenizer.tokenize_words(text)
        elif tokenize_method == "sentence":
            tokens = self.tokenizer.tokenize_sentences(text)
        else:  # regexp
            tokens = self.tokenizer.tokenize_regexp(text)
        
        return TextAnalyzer(tokens, name)
    
    def analyze_corpus(self, file_id: str = "austen-emma.txt") -> Dict:
        """
        말뭉치 전체 분석 수행
        
        Args:
            file_id: 말뭉치 파일 ID
            
        Returns:
            분석 결과 딕셔너리
        """
        # 말뭉치 로드
        raw_text = self.corpus_manager.load_corpus(file_id)
        
        # 토큰화
        tokens = self.tokenizer.tokenize_regexp(raw_text)
        
        # 텍스트 분석기 생성
        analyzer = TextAnalyzer(tokens, name=file_id)
        
        # 분석 수행
        freq_dist = analyzer.get_freq_dist()
        most_common = analyzer.most_common(10)
        
        return {
            "file_id": file_id,
            "total_tokens": len(tokens),
            "unique_tokens": len(freq_dist),
            "most_common_words": most_common,
            "analyzer": analyzer,
            "freq_dist": freq_dist
        }


# 사용 예제 (스크립트로 실행 시)
if __name__ == "__main__":
    # NLP 서비스 초기화
    nlp_service = NLPService()
    
    # Emma 말뭉치 분석
    print("=" * 50)
    print("Emma 말뭉치 분석")
    print("=" * 50)
    
    result = nlp_service.analyze_corpus("austen-emma.txt")
    
    print(f"\n파일: {result['file_id']}")
    print(f"전체 토큰 수: {result['total_tokens']:,}")
    print(f"고유 토큰 수: {result['unique_tokens']:,}")
    print(f"\n가장 빈번한 단어 Top 10:")
    for word, count in result['most_common_words']:
        print(f"  {word}: {count:,}")
    
    # 고유명사 추출
    stopwords = ["Mr.", "Mrs.", "Miss", "Mr", "Mrs", "Dear"]
    proper_nouns_fd = result['analyzer'].filter_proper_nouns(stopwords)
    
    print(f"\n고유명사 Top 5:")
    for word, count in proper_nouns_fd.most_common(5):
        print(f"  {word}: {count:,}")
    
    # 워드클라우드 생성 및 저장
    print("\n워드클라우드 생성 중...")
    
    # save 폴더 경로 설정
    current_dir = Path(__file__).parent
    save_dir = current_dir.parent / "save"
    save_dir.mkdir(exist_ok=True)
    
    # 워드클라우드 생성
    wc = nlp_service.visualizer.generate_wordcloud(
        proper_nouns_fd, 
        width=1200, 
        height=800,
        background_color="white",
        show=False
    )
    
    # 이미지 파일로 저장
    output_path = save_dir / "emma_wordcloud.png"
    wc.to_file(str(output_path))
    print(f"워드클라우드 저장 완료: {output_path}")
    
    # matplotlib으로도 저장 (고해상도)
    plt.figure(figsize=(12, 8), dpi=300)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    output_path_high = save_dir / "emma_wordcloud_high_res.png"
    plt.savefig(str(output_path_high), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"고해상도 워드클라우드 저장 완료: {output_path_high}")
    
    print("\n완료!")