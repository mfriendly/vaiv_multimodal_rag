# 🎯 RTRAG Milvus - Multimodal RAG System

# 1. 재난 매뉴얼 DB 생성python build_text_db.py \  --input data/manuals/disaster_manuals.json \  --collection disaster_manual \  --db-file db/fire.db# 2. 뉴스 + 매뉴얼 동시 검색 (다중 컬렉션!)python run_search.py \  --mode text \  --query "화재 대피 요령" \  --collection fire_multimodal_demo,disaster_manual \  --db-file db/fire.db# 3. 매뉴얼만 검색 (topic 필터)python run_search.py \  --mode text \  --query "지진 발생 시 행동" \  --collection disaster_manual \  --db-file db/fire.db \  --topic earthquake

FAISS에서 Milvus로 마이그레이션한 멀티모달 RAG 시스템입니다. 텍스트와 이미지를 함께 처리하여 더 풍부한 검색 경험을 제공합니다.

## 📚 목차

- [개요](#개요)
- [빠른 시작](#빠른-시작)
- [파일 구조](#파일-구조)
- [설치 방법](#설치-방법)
- [사용 가이드](#사용-가이드)
- [데모 실행](#데모-실행)

---

## 🌟 개요

### 주요 특징

✨ **두 가지 RAG 옵션**
- **텍스트 전용 RAG**: Milvus Lite 기반, Docker 불필요
- **멀티모달 RAG**: 텍스트 + 이미지, 고급 검색 기능

🎨 **Multimodal 지원**
- 텍스트 임베딩 (ko-sroberta-multitask, 768차원)
- 이미지 임베딩 (CLIP, 512차원)
- 하이브리드 검색 (텍스트 + 이미지)

🔍 **강력한 검색 기능**
- 메타데이터 필터링 (카테고리, 토픽, 날짜)
- 벡터 유사도 검색
- 복합 조건 쿼리

---

## 🚀 빠른 시작

### 옵션 1: 멀티모달 RAG 데모 (추천) ⭐

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 멀티모달 데모 실행 (자동 DB 생성 + 검색 데모)
bash demo_fire_multimodal.sh
```

**이 스크립트가 자동으로 수행하는 작업:**
- 화재 뉴스와 이미지 자동 매칭 (파일명 = doc_id)
- 멀티모달 DB 생성
- 텍스트/이미지/하이브리드 검색 데모

### 옵션 2: 수동 멀티모달 DB 생성

```bash
# 클러스터된 뉴스만 사용
python create_multimodal_db_from_images.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images naver_news_images/fire \
  --collection fire_multimodal \
  --news-range fire_clustered \
  --clustered-csv clustered_news.csv

# 검색 예제
python demo_multimodal_fire.py \
  --mode text \
  --collection fire_multimodal \
  --query "화재 사건"
```

### 옵션 3: 텍스트 전용 RAG

```bash
# 1. 데이터 변환
python convert_news_to_milvus_lite_v2.py \
  --input news_data/01_disaster_Fire_3years.json \
  --collection fire_news

# 2. 검색
python milvus_lite_search_v2.py \
  --collection fire_news \
  --query "화재 사건" \
  --top-k 5
```

---

## 📁 파일 구조

```
RTRAG_milvus_share/
├── README.md                              # 이 파일
├── requirements.txt                       # Python 패키지 목록
│
├── 🎯 멀티모달 RAG (추천)
│   ├── demo_fire_multimodal.sh                 # 🌟 원클릭 데모 스크립트
│   ├── demo_multimodal_fire.py                 # 검색 데모 (텍스트/이미지/하이브리드)
│   ├── create_multimodal_db_from_images.py    # 멀티모달 DB 생성 (파일명=doc_id)
│   └── download_free_images.py                 # 무료 이미지 다운로드
│
├── 🔧 텍스트 전용 RAG
│   ├── convert_news_to_milvus_lite_v2.py      # 뉴스 → Milvus Lite 변환
│   └── milvus_lite_search_v2.py               # Milvus Lite 검색
│
├── 📊 데이터
│   ├── news_data/                              # 뉴스 JSON 데이터
│   │   └── 01_disaster_Fire_3years.json
│   ├── naver_news_images/fire/                 # 뉴스 이미지 (파일명=doc_id)
│   ├── query_image_data/fire/                  # 검색용 쿼리 이미지
│   └── clustered_news.csv                      # 클러스터링된 뉴스 목록
│
└── 🛠️ 유틸리티
    ├── rename_images.py                        # 이미지 일괄 리네이밍
    └── manual_image_matcher.py                 # 뉴스-이미지 수동 매칭
```

---

## 🛠️ 설치 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 뉴스 데이터 (이미 포함됨)
ls news_data/

# 이미지 데이터 (이미 포함됨)
ls image_data/
```

---

## 📖 사용 가이드

### 1️⃣ 멀티모달 RAG (추천)

#### 🌟 원클릭 데모 실행
```bash
bash demo_fire_multimodal.sh
```

**데모 스크립트가 자동으로 수행하는 작업:**
1. 이미지 파일명(doc_id)으로 뉴스와 자동 매칭
2. 멀티모달 DB 생성 (클러스터된 뉴스 사용)
3. 텍스트/이미지/하이브리드 검색 시연

#### 수동 DB 생성 및 검색

**1. 멀티모달 DB 생성**
```bash
# 전체 뉴스 사용
python create_multimodal_db_from_images.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images naver_news_images/fire \
  --collection fire_multimodal \
  --news-range fire_all

# 클러스터된 뉴스만 사용 (추천)
python create_multimodal_db_from_images.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images naver_news_images/fire \
  --collection fire_multimodal \
  --news-range fire_clustered \
  --clustered-csv clustered_news.csv
```

**2. 텍스트 검색**
```bash
python demo_multimodal_fire.py \
  --mode text \
  --collection fire_multimodal \
  --query "화재 사건" \
  --top-k 5
```

**3. 이미지 검색**
```bash
python demo_multimodal_fire.py \
  --mode image \
  --collection fire_multimodal \
  --image query_image_data/fire/fire1.jpg \
  --top-k 5
```

**4. 하이브리드 검색**
```bash
python demo_multimodal_fire.py \
  --mode hybrid \
  --collection fire_multimodal \
  --query "대형 화재" \
  --image query_image_data/fire/fire2.jpg \
  --top-k 5
```

### 2️⃣ 텍스트 전용 RAG

#### Milvus Lite (Docker 불필요)
```bash
# 데이터 변환
python convert_news_to_milvus_lite_v2.py \
  --input news_data/01_disaster_Fire_3years.json \
  --collection fire_news \
  --db-file ./fire_news.db

# 검색
python milvus_lite_search_v2.py \
  --collection fire_news \
  --query "화재 사건" \
  --top-k 5 \
  --db-file ./fire_news.db
```

#### Milvus Server (Docker 필요)
```bash
# Milvus Server 시작 (Docker)
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest

# 데이터 변환
python convert_news_to_milvus.py \
  --input news_data/01_disaster_Fire_3years.json \
  --collection fire_news

# 검색
python milvus_rag_search.py \
  --collection fire_news \
  --query "화재 사건" \
  --top-k 5
```

### 3️⃣ 이미지 관리 (유틸리티)

#### 수동 이미지 매칭 (대화형 모드)
```bash
# 대화형 모드로 특정 뉴스에 특정 이미지 매칭
python manual_image_matcher.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images naver_news_images/fire

# 대화형 모드 명령어:
#   n [검색어]  - 뉴스 목록 보기
#   i           - 이미지 목록 보기
#   m           - 현재 매핑 목록 보기
#   a           - 새 매핑 추가
#   r           - 매핑 제거
#   s <file>    - 파일로 저장
#   q           - 종료
```

#### CLI 모드로 직접 매칭
```bash
# 직접 매칭 추가
python manual_image_matcher.py \
  --add "202304110010013873784:fire1.jpg" \
  --add "202304110010013872301:fire3.jpg" \
  --output manual_mappings.json
```

#### 무료 이미지 다운로드
```bash
# Unsplash에서 무료 이미지 다운로드
python download_free_images.py \
  --source unsplash \
  --api-key YOUR_UNSPLASH_KEY \
  --query "fire disaster emergency" \
  --output naver_news_images/fire_downloaded \
  --limit 20
```

---

## 🎯 데모 실행

### 멀티모달 RAG 데모

```bash
# 🌟 원클릭 데모 실행 (추천)
bash demo_fire_multimodal.sh
```

이 데모는 다음을 자동으로 수행합니다:

**1단계: DB 생성**
- `news_data/01_disaster_Fire_3years.json`에서 뉴스 로드
- `clustered_news.csv`로 클러스터된 뉴스 필터링
- `naver_news_images/fire/`의 이미지 자동 매칭 (파일명 = doc_id)
- 멀티모달 컬렉션 생성

**2단계: 검색 시연**
- 텍스트 검색: "화재 사건", "대형 화재 진압"
- 이미지 검색: `query_image_data/fire/fire1.jpg`로 검색
- 하이브리드 검색: 텍스트 + 이미지 결합

### 데모 설정 변경

`demo_fire_multimodal.sh` 파일에서 설정을 변경할 수 있습니다:

```bash
NEWS_RANGE="fire_clustered"  # fire_all 또는 fire_clustered
COLLECTION_NAME="fire_multimodal_demo"
DB_FILE="./fire_multimodal_demo.db"
```

### 생성되는 파일

데모 실행 후:
- `fire_multimodal_demo.db` - Milvus Lite 데이터베이스

---

## 💻 API 문서

### 멀티모달 RAG API

```python
from multimodal_rag_v2 import MultimodalRAG

# 컬렉션 생성
rag = MultimodalRAG(
    collection_name="fire_news",
    db_file="./demo.db"
)

# 텍스트 검색
results = rag.search_by_text("화재 사건", top_k=5)

# 이미지 검색
results = rag.search_by_image("path/to/image.jpg", top_k=5)

# 하이브리드 검색
results = rag.search_hybrid(
    query="화재",
    image_path="path/to/image.jpg",
    top_k=5
)
```

### 텍스트 전용 RAG API

```python
from milvus_lite_search_v2 import MilvusLiteRAG

# 검색 시스템 초기화
rag = MilvusLiteRAG(
    collection_name="fire_news",
    db_file="./fire_news.db"
)

# 텍스트 검색
results = rag.search_by_text("화재 사건", top_k=5)

# 메타데이터 필터링
results = rag.search_with_filter(
    query="화재",
    category="disaster",
    topic="fire",
    top_k=10
)
```

---

## 🎓 고급 기능

### 메타데이터 필터링

```python
# 카테고리별 검색
results = rag.search_by_text(
    query="화재",
    category="disaster",
    top_k=10
)

# 날짜 범위 검색
results = rag.search_by_text(
    query="화재",
    date_start="20220101",
    date_end="20231231",
    top_k=10
)
```

### 배치 처리

```bash
# 여러 파일 처리
python convert_news_to_milvus_lite_v2.py \
  --input news_data/ \
  --collection all_news \
  --db-file ./all_news.db
```

---

## 📊 성능 비교

| 기능 | 텍스트 전용 | 멀티모달 |
|------|-------------|----------|
| **설치 복잡도** | 낮음 | 중간 |
| **검색 정확도** | 좋음 | 매우 좋음 |
| **이미지 지원** | ❌ | ✅ |
| **하이브리드 검색** | ❌ | ✅ |
| **메모리 사용량** | 낮음 | 중간 |
| **추천 용도** | 간단한 검색 | 고급 검색 |

---

## 🆘 도움말

### 문서
- 📘 [README_MILVUS_MIGRATION.md](README_MILVUS_MIGRATION.md) - 상세 마이그레이션 가이드
- 📙 [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) - 파이프라인 아키텍처
- 🖼️ [IMAGE_MANAGEMENT_GUIDE.md](IMAGE_MANAGEMENT_GUIDE.md) - 이미지 관리 완벽 가이드

### 외부 리소스
- [Milvus Documentation](https://milvus.io/docs)
- [Milvus Lite](https://github.com/milvus-io/milvus-lite)
- [LangChain Milvus](https://python.langchain.com/docs/integrations/vectorstores/milvus)

---

## ✅ 체크리스트

### 시작하기
- [ ] Python 패키지 설치
- [ ] 뉴스 데이터 확인
- [ ] 이미지 데이터 확인

### 기본 기능
- [ ] 멀티모달 데모 실행
- [ ] 텍스트 검색 테스트
- [ ] 이미지 검색 테스트
- [ ] 하이브리드 검색 테스트

### 고급 기능 (선택)
- [ ] 무료 이미지 다운로드
- [ ] 스마트 이미지 매칭
- [ ] 메타데이터 필터링

---

## 🎯 권장 사항

**빠른 시작:**
- ✅ 멀티모달 데모 실행: `bash demo_fire_multimodal.sh`
- ✅ 텍스트 전용: `convert_news_to_milvus_lite_v2.py` 사용

**고급 사용:**
- ✅ 무료 이미지 다운로드 후 스마트 매칭
- ✅ 메타데이터 필터링 활용
- ✅ 하이브리드 검색 최적화

---

## 📝 라이센스

이 프로젝트는 기존 RTRAG 프로젝트의 일부입니다.

---

## 🙏 감사의 말

- [Milvus](https://milvus.io/) - 벡터 데이터베이스
- [LangChain](https://www.langchain.com/) - RAG 프레임워크
- [HuggingFace](https://huggingface.co/) - 임베딩 모델
- [CLIP](https://openai.com/research/clip) - 멀티모달 모델

---

**Happy RAG Building! 🚀**