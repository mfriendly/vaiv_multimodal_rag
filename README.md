# RTRAG Milvus - Multimodal RAG System

텍스트 + 이미지 멀티모달 검색을 지원하는 Milvus 기반 RAG 시스템

## Quick Start

```bash
pip install -r requirements.txt
bash demo_fire_multimodal.sh
```

## Usage

### 1. DB 생성

```bash
# 멀티모달 DB (텍스트 + 이미지)
python create_multimodal_db_from_images.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images naver_news_images/fire \
  --collection fire_multimodal \
  --news-range fire_clustered \
  --clustered-csv clustered_news.csv \
  --db-file ./fire.db
```

### 2. 검색

```bash
# 텍스트 검색
python demo_multimodal_fire.py \
  --mode text \
  --collection fire_multimodal \
  --query "화재 사건" \
  --db-file ./fire.db

# 이미지 검색
python demo_multimodal_fire.py \
  --mode image \
  --collection fire_multimodal \
  --image query_image_data/fire/fire1.jpg \
  --db-file ./fire.db

# 하이브리드 검색 (텍스트 + 이미지)
python demo_multimodal_fire.py \
  --mode hybrid \
  --collection fire_multimodal \
  --query "대형 화재" \
  --image query_image_data/fire/fire2.jpg \
  --db-file ./fire.db
```

### 3. 메타데이터 필터링

```bash
python demo_multimodal_fire.py \
  --mode text \
  --collection fire_multimodal \
  --query "화재" \
  --date-start 20220101 \
  --date-end 20231231 \
  --category disaster \
  --topic fire \
  --db-file ./fire.db
```

## Project Structure

```
├── demo_fire_multimodal.sh          # 원클릭 데모
├── create_multimodal_db_from_images.py  # DB 생성
├── demo_multimodal_fire.py          # 검색 실행
├── news_data/                       # 뉴스 JSON
├── naver_news_images/               # 뉴스 이미지 (파일명=doc_id)
├── query_image_data/                # 쿼리 이미지
└── clustered_news.csv               # 클러스터링된 뉴스 목록
```

## API

```python
from demo_multimodal_fire import MultimodalSearcher

searcher = MultimodalSearcher(db_file="./fire.db", collection_name="fire_multimodal")

# 텍스트 검색
results = searcher.search_by_text("화재 사건", top_k=5)

# 이미지 검색
results = searcher.search_by_image("query.jpg", top_k=5)
```

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt`
