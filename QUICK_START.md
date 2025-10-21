# 🚀 빠른 시작 가이드

RTRAG Milvus 시스템을 빠르게 시작하는 방법입니다.

---

## 📦 1. 설치

```bash
cd /mnt/nvme02/home/tdrag/vaiv/RTRAG_milvus_share
pip install -r requirements.txt
```

---

## 🎯 2. 이미지 준비 (최초 1회)

### 옵션 A: 기존 이미지 리네이밍

```bash
# 미리보기
python rename_images.py --input image_data/fire --dry-run

# 실행 (자동 백업 생성)
python rename_images.py --input image_data/fire
```

결과: `fire1.jpg`, `fire2.jpg`, ..., `fireN.jpg`

### 옵션 B: 새 이미지 다운로드

```bash
export UNSPLASH_KEY="your_key"

python download_free_images.py \
  --source unsplash \
  --api-key $UNSPLASH_KEY \
  --query "fire disaster emergency" \
  --output image_data/fire \
  --limit 20

python rename_images.py --input image_data/fire
```

---

## 🔗 3. 뉴스-이미지 매칭

### 옵션 A: 자동 데모 (랜덤 매칭)

```bash
bash demo_fire_multimodal.sh
```

### 옵션 B: 수동 매칭 (추천)

```bash
python manual_image_matcher.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images image_data/fire
```

**대화형 명령어:**
- `n` - 뉴스 목록
- `i` - 이미지 목록  
- `a` - 매칭 추가
- `m` - 현재 매칭 보기
- `s manual_mappings.json` - 저장
- `q` - 종료

---

## 🔍 4. RAG 시스템 사용

### 멀티모달 RAG

```bash
# 1. 컬렉션 생성
python multimodal_rag_v2.py \
  --mode create \
  --collection fire_multimodal \
  --input news_data/01_disaster_Fire_3years.json \
  --images manual_mappings.json

# 2. 텍스트 검색
python multimodal_rag_v2.py \
  --mode search \
  --collection fire_multimodal \
  --query "화재 사건" \
  --top-k 5

# 3. 이미지 검색
python multimodal_rag_v2.py \
  --mode search-image \
  --collection fire_multimodal \
  --image image_data/fire/fire1.jpg \
  --top-k 5

# 4. 하이브리드 검색
python multimodal_rag_v2.py \
  --mode hybrid \
  --collection fire_multimodal \
  --query "화재" \
  --image image_data/fire/fire2.jpg \
  --top-k 5
```

### 텍스트 전용 RAG

```bash
# 1. 데이터 변환
python convert_news_to_milvus_lite_v2.py \
  --input news_data/01_disaster_Fire_3years.json \
  --collection fire_news \
  --db-file ./fire_news.db

# 2. 검색
python milvus_lite_search_v2.py \
  --collection fire_news \
  --query "화재 사건" \
  --top-k 5 \
  --db-file ./fire_news.db
```

---

## 📚 더 알아보기

- 📘 [README.md](README.md) - 전체 문서
- 🖼️ [IMAGE_MANAGEMENT_GUIDE.md](IMAGE_MANAGEMENT_GUIDE.md) - 이미지 관리
- 📙 [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) - 아키텍처

---

## 💡 자주 사용하는 명령어

```bash
# 이미지 리네이밍
python rename_images.py --input image_data/fire

# 수동 매칭
python manual_image_matcher.py --news NEWS_FILE --images IMAGE_DIR

# 빠른 데모
bash demo_fire_multimodal.sh

# CLI 매칭
python manual_image_matcher.py --add "doc_id:fire1.jpg" --output mappings.json
```

---

**Happy RAG Building! 🚀**

