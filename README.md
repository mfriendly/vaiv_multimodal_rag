# 🎯 RTRAG Milvus - Multimodal RAG System

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

# 2. 멀티모달 데모 실행
bash demo_fire_multimodal.sh
```

### 옵션 2: 텍스트 전용 RAG

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
├── README_MILVUS_MIGRATION.md             # 상세 마이그레이션 가이드
├── PIPELINE_ARCHITECTURE.md               # 파이프라인 아키텍처
├── requirements.txt                       # Python 패키지 목록
├── LICENSE                                # 라이센스
│
├── 🎯 멀티모달 RAG (추천)
│   ├── demo_fire_multimodal.sh           # 멀티모달 데모 실행
│   ├── demo_multimodal_fire.py           # 뉴스 데이터 준비
│   ├── multimodal_rag_v2.py               # 멀티모달 RAG 시스템
│   ├── add_images_to_milvus.py           # 이미지 추가
│   └── download_free_images.py           # 무료 이미지 다운로드
│
├── 🔧 텍스트 전용 RAG
│   ├── convert_news_to_milvus_lite_v2.py # 뉴스 → Milvus Lite 변환
│   ├── milvus_lite_search_v2.py          # Milvus Lite 검색
│   └── milvus_rag_search.py              # Milvus Server 검색
│
├── 📊 데이터
│   ├── news_data/                         # 뉴스 데이터
│   └── image_data/                        # 이미지 데이터
│
└── 🛠️ 유틸리티
    ├── run_metadata_v2.py                # 메타데이터 처리
    ├── rename_images.py                  # 이미지 일괄 리네이밍 (fire1.jpg ~ fireN.jpg)
    └── manual_image_matcher.py           # 뉴스-이미지 수동 매칭
```

---

## 🛠️ 설치 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
* 뉴스 데이터 다운로드 [바로가기](https://drive.google.com/drive/folders/1gTBjmM6WwJcSsrGyEl1Cl7t_OG5CVi6s?usp=drive_link)
  * `news_data`에 저장
* 이미지 데이터 다운로드 [바로가기](https://drive.google.com/drive/folders/1ik4d0H5QMBTW2ykWtDh-53aKvwxLtAan?usp=drive_link)
  * `image_data`에 저장
```bash
# 뉴스 데이터 (이미 포함됨)
ls news_data/

# 이미지 데이터 (이미 포함됨)
ls image_data/
```

---

## 📖 사용 가이드

### 1️⃣ 멀티모달 RAG (추천)

#### 전체 데모 실행
```bash
bash demo_fire_multimodal.sh
```

#### 단계별 실행
```bash
# 1. 데이터 준비
python demo_multimodal_fire.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images image_data/fire \
  --limit 100 \
  --ratio 0.3

# 2. 멀티모달 컬렉션 생성
python multimodal_rag_v2.py \
  --mode create \
  --collection fire_multimodal \
  --input prepared_fire_news.json \
  --images fire_image_mappings.json

# 3. 텍스트 검색
python multimodal_rag_v2.py \
  --mode search \
  --collection fire_multimodal \
  --query "화재 사건" \
  --top-k 5

# 4. 이미지 검색
python multimodal_rag_v2.py \
  --mode search-image \
  --collection fire_multimodal \
  --image image_data/fire/fire1.jpg \
  --top-k 5

# 5. 하이브리드 검색
python multimodal_rag_v2.py \
  --mode hybrid \
  --collection fire_multimodal \
  --query "화재" \
  --image image_data/fire/fire2.jpg \
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

### 3️⃣ 이미지 관리

#### 이미지 리네이밍 (fire1.jpg ~ fireN.jpg)
```bash
# 미리보기 (실제 변경 없음)
python rename_images.py --input image_data/fire --dry-run

# 실제 리네이밍 (자동 백업 생성)
python rename_images.py --input image_data/fire

# 커스텀 설정
python rename_images.py --input image_data/fire --prefix fire --start 1 --ext .jpg
```

#### 수동 이미지 매칭 (대화형 모드)
```bash
# 대화형 모드로 특정 뉴스에 특정 이미지 매칭
python manual_image_matcher.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images image_data/fire

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
  --add "fire_news_001:fire1.jpg" \
  --add "fire_news_002:fire3.jpg" \
  --output manual_mappings.json

# 기존 매핑 수정
python manual_image_matcher.py \
  --edit manual_mappings.json \
  --add "fire_news_003:fire5.jpg" \
  --output manual_mappings_updated.json
```

#### 무료 이미지 다운로드
```bash
# 무료 이미지 다운로드
python download_free_images.py \
  --source unsplash \
  --api-key YOUR_UNSPLASH_KEY \
  --query "fire disaster emergency" \
  --output image_data/fire_downloaded \
  --limit 20

# 다운로드 후 리네이밍
python rename_images.py --input image_data/fire_downloaded
```

#### 기존 컬렉션에 이미지 추가
```bash
python add_images_to_milvus.py \
  --collection fire_news \
  --images manual_mappings.json
```

---

## 🎯 데모 실행

### 멀티모달 RAG 데모

```bash
# 전체 데모 실행 (추천)
bash demo_fire_multimodal.sh
```

이 데모는 다음을 수행합니다:
1. 화재 뉴스 데이터 로드
2. 이미지 랜덤 할당
3. 멀티모달 컬렉션 생성
4. 다양한 검색 방법 시연:
   - 텍스트 검색
   - 이미지 검색
   - 하이브리드 검색

### 결과 파일

데모 실행 후 생성되는 파일들:
- `prepared_fire_news.json` - 준비된 뉴스 데이터
- `fire_image_mappings.json` - 이미지 매핑
- `multimodal_demo.db` - Milvus 데이터베이스

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