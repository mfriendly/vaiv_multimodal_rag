# Milvus Multimodal RAG Migration Guide

FAISS에서 Milvus로 마이그레이션하고, 향후 이미지를 포함한 Multimodal RAG를 지원하는 시스템입니다.

## 📋 목차

1. [왜 Milvus인가?](#왜-milvus인가)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [설치 및 설정](#설치-및-설정)
4. [사용법](#사용법)
5. [Multimodal RAG 확장](#multimodal-rag-확장)
6. [API 레퍼런스](#api-레퍼런스)

---

## 왜 Milvus인가?

### FAISS vs Milvus 비교

| 기능 | FAISS | Milvus |
|------|-------|--------|
| **확장성** | 단일 머신 제한 | 분산 시스템 지원 |
| **메타데이터 필터링** | 제한적 | 강력한 스칼라 필터링 |
| **실시간 업데이트** | 어려움 | 지원 |
| **Multimodal** | 복잡한 구현 필요 | 여러 벡터 필드 지원 |
| **운영 관리** | 직접 관리 | 웹 UI, 모니터링 제공 |
| **성능** | 빠름 (인메모리) | 빠름 (디스크+인메모리) |

### Milvus 선택 이유

1. **대규모 데이터 처리**: 수백만~수억 개의 벡터 처리 가능
2. **메타데이터 기반 필터링**: 날짜, 카테고리, 토픽 등으로 효율적 필터링
3. **Multimodal 지원**: 텍스트와 이미지 임베딩을 동시에 저장하고 검색
4. **실시간 업데이트**: 새로운 뉴스를 실시간으로 추가 가능
5. **프로덕션 준비**: 모니터링, 백업, 복구 등 기업용 기능

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Milvus Collection Schema                 │
├─────────────────────────────────────────────────────────────┤
│  - id (Primary Key)                                          │
│  - text_embedding (Float Vector, dim=768)    ← 텍스트 검색   │
│  - image_embedding (Float Vector, dim=512)   ← 이미지 검색   │
│  - doc_id, title, content, date, url, source                │
│  - category, topic                           ← 필터링        │
│  - has_image, image_url, image_caption       ← Multimodal   │
│  - created_at                                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  News JSON ──────────┐                                       │
│                      │                                       │
│  Images (optional) ──┼──> NewsToMilvusConverter            │
│                      │    - Text Embedding (ko-sroberta)    │
│                      │    - Image Embedding (CLIP)          │
│                      │    - Metadata Extraction (GPT)       │
│                      │                                       │
│                      └──> Milvus Collection                 │
│                           ├─ IVF_FLAT Index (text)          │
│                           └─ IVF_FLAT Index (image)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Search Modes                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Text Search:    Query → Text Embedding → Milvus        │
│  2. Image Search:   Image → CLIP Embedding → Milvus        │
│  3. Hybrid Search:  Text + Image → Weighted Fusion         │
│  4. Filtered Search: Query + Metadata Filter → Milvus      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 설치 및 설정

### 1. Milvus 설치 (Docker 사용)

```bash
# Milvus Standalone 설치
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d

# Milvus가 실행되었는지 확인
docker-compose ps

# Attu (Milvus Web UI) 설치 (선택사항)
docker run -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:latest
```

Milvus는 다음 포트에서 실행됩니다:
- `19530`: Milvus gRPC 서비스
- `9091`: Milvus metrics
- `8000`: Attu Web UI (선택사항)

### 2. Python 패키지 설치

```bash
# 기본 패키지
pip install pymilvus
pip install langchain langchain-community
pip install sentence-transformers
pip install openai

# 이미지 처리용 (Multimodal RAG)
pip install transformers torch torchvision
pip install Pillow requests

# 기타 유틸리티
pip install tqdm
```

### 3. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
EOF

# 환경 변수 로드
export $(cat .env | xargs)
```

---

## 사용법

### 1️⃣ 뉴스 데이터를 Milvus로 변환

#### 단일 파일 처리

```bash
python convert_news_to_milvus.py \
  --input news_data/01_disaster_Fire_3years.json \
  --collection fire_news \
  --milvus-host localhost \
  --milvus-port 19530
```

#### 디렉토리 전체 처리

```bash
python convert_news_to_milvus.py \
  --input news_data/ \
  --collection disaster_news \
  --all
```

#### GPT 메타데이터 추출 사용

```bash
python convert_news_to_milvus.py \
  --input news_data/01_disaster_Fire_3years.json \
  --collection fire_news \
  --openai-key $OPENAI_API_KEY
```

#### GPT 없이 파일명 기반 메타데이터만 사용

```bash
python convert_news_to_milvus.py \
  --input news_data/01_disaster_Fire_3years.json \
  --collection fire_news \
  --no-gpt
```

### 2️⃣ Milvus에서 검색

#### 텍스트 검색

```bash
# 기본 검색
python milvus_rag_search.py \
  --collection fire_news \
  --query "대형 화재 사건" \
  --mode text \
  --top-k 10

# 카테고리 필터링
python milvus_rag_search.py \
  --collection fire_news \
  --query "화재" \
  --category disaster \
  --topic fire \
  --top-k 5
```

#### 날짜 범위 필터링

```bash
python milvus_rag_search.py \
  --collection fire_news \
  --query "화재 사건" \
  --date-start 20230101 \
  --date-end 20231231 \
  --top-k 10
```

#### 결과를 JSON으로 저장

```bash
python milvus_rag_search.py \
  --collection fire_news \
  --query "화재" \
  --output search_results.json
```

### 3️⃣ 이미지 추가 (Multimodal RAG)

#### 이미지 매핑 JSON 생성

```json
[
  {
    "doc_id": "fire_news_001",
    "image_url": "https://example.com/fire_image1.jpg",
    "caption": "서울시 강남구 건물 화재 현장"
  },
  {
    "doc_id": "fire_news_002",
    "image_path": "/path/to/local/image.jpg",
    "caption": "소방관들의 진화 작업"
  }
]
```

#### 이미지 추가 실행

```bash
python add_images_to_milvus.py \
  --collection fire_news \
  --images image_mappings.json
```

### 4️⃣ Multimodal 검색

#### 이미지로 검색

```bash
python milvus_rag_search.py \
  --collection fire_news \
  --image path/to/fire_image.jpg \
  --mode image \
  --top-k 5
```

#### 하이브리드 검색 (텍스트 + 이미지)

```bash
python milvus_rag_search.py \
  --collection fire_news \
  --query "화재 사고" \
  --image path/to/fire_image.jpg \
  --mode hybrid \
  --top-k 10
```

---

## Multimodal RAG 확장

### 현재 구현 상태

✅ **완료된 기능:**
- 텍스트 임베딩 및 검색 (ko-sroberta-multitask)
- 메타데이터 기반 필터링 (날짜, 카테고리, 토픽)
- 이미지 임베딩을 위한 스키마 설계
- CLIP 기반 이미지 검색 인프라

🚧 **향후 추가 예정:**
- BLIP 기반 이미지 캡셔닝
- Vision Transformer를 활용한 고급 이미지 분석
- 텍스트-이미지 크로스 모달 검색 최적화
- 비디오 프레임 추출 및 검색

### 이미지 추가 워크플로우

```python
# 예제: 프로그래밍 방식으로 이미지 추가
from add_images_to_milvus import ImageToMilvusAdder

adder = ImageToMilvusAdder(
    collection_name="fire_news",
    milvus_host="localhost",
    milvus_port="19530"
)

# 단일 이미지 추가
adder.update_document_with_image(
    doc_id="fire_news_001",
    image_source="https://example.com/image.jpg",
    image_caption="화재 현장 사진"
)

# 배치 추가
image_mappings = [
    {"doc_id": "news_001", "image_url": "...", "caption": "..."},
    {"doc_id": "news_002", "image_url": "...", "caption": "..."},
]
adder.batch_add_images(image_mappings)
```

### 검색 예제

```python
# 예제: Python API 사용
from milvus_rag_search import MilvusMultimodalRAG

rag = MilvusMultimodalRAG(collection_name="fire_news")

# 텍스트 검색
results = rag.search_by_text(
    query="대형 화재",
    top_k=5,
    filter_expr='category == "disaster" && date >= "20230101"'
)

# 이미지 검색
results = rag.search_by_image(
    image_path="query_image.jpg",
    top_k=5
)

# 하이브리드 검색
results = rag.hybrid_search(
    text_query="화재 사고",
    image_path="query_image.jpg",
    text_weight=0.6,
    image_weight=0.4,
    top_k=10
)

# 결과 출력
for result in results:
    print(f"Title: {result['title']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:100]}...")
```

---

## API 레퍼런스

### NewsToMilvusConverter

뉴스 데이터를 Milvus로 변환하는 메인 클래스

```python
converter = NewsToMilvusConverter(
    milvus_host="localhost",
    milvus_port="19530",
    openai_api_key="your_key",
    text_embedding_dim=768,
    image_embedding_dim=512
)

# 단일 파일 처리
converter.process_single_file(
    input_file="news.json",
    use_gpt_metadata=True
)

# 디렉토리 처리
converter.process_directory(
    input_dir="news_data/",
    use_gpt_metadata=True
)
```

### MilvusMultimodalRAG

Milvus에서 검색하는 클래스

```python
rag = MilvusMultimodalRAG(
    collection_name="fire_news",
    milvus_host="localhost",
    milvus_port="19530"
)

# 텍스트 검색
results = rag.search_by_text(
    query="검색어",
    top_k=10,
    filter_expr='category == "disaster"',
    output_fields=["title", "content", "date"]
)

# 메타데이터 필터 검색
results = rag.search_with_metadata_filter(
    query="검색어",
    category="disaster",
    topic="fire",
    date_start="20230101",
    date_end="20231231",
    top_k=10
)
```

### ImageToMilvusAdder

이미지를 기존 컬렉션에 추가하는 클래스

```python
adder = ImageToMilvusAdder(
    collection_name="fire_news",
    milvus_host="localhost",
    clip_model="openai/clip-vit-base-patch32"
)

# 배치 추가
adder.batch_add_images(image_mappings)
```

---

## 성능 최적화

### 인덱스 파라미터 튜닝

```python
# IVF_FLAT 인덱스 파라미터
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}  # 클러스터 수 (데이터 크기에 따라 조정)
}

# 검색 파라미터
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10}  # 검색할 클러스터 수 (정확도 vs 속도)
}
```

### 대용량 데이터 처리

```python
# 배치 사이즈 조정
converter.process_single_file(
    input_file="large_news.json",
    batch_size=1000  # 임베딩 배치 크기
)

# Milvus 삽입 배치 크기
converter.insert_to_milvus(
    collection=collection,
    data=data,
    batch_size=5000  # 삽입 배치 크기
)
```

---

## 트러블슈팅

### 1. Milvus 연결 실패

```bash
# Milvus 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs milvus-standalone

# 재시작
docker-compose restart
```

### 2. CUDA 메모리 부족

```python
# CPU 사용으로 변경
model_kwargs={'device': 'cpu'}

# 배치 크기 줄이기
batch_size=50
```

### 3. 임베딩 모델 다운로드 실패

```bash
# 수동 다운로드 및 캐시
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"
```

---

## 다음 단계

1. **프로덕션 배포**
   - Milvus 클러스터 구성
   - 로드 밸런싱 설정
   - 모니터링 및 알림 설정

2. **고급 기능 추가**
   - 다국어 지원 (다중 임베딩 모델)
   - 실시간 스트리밍 인덱싱
   - A/B 테스트 프레임워크

3. **Multimodal 확장**
   - 비디오 프레임 분석
   - 오디오 임베딩 추가
   - 멀티모달 퓨전 전략 개선

---

## 라이센스 및 참고자료

- [Milvus Documentation](https://milvus.io/docs)
- [LangChain Milvus Integration](https://python.langchain.com/docs/integrations/vectorstores/milvus)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Korean SRoBERTa](https://huggingface.co/jhgan/ko-sroberta-multitask)

---

## 지원 및 문의

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.

**Happy RAG Building! 🚀**

