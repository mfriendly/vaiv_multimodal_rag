# 🖼️ 이미지 관리 가이드

이미지 파일 관리 및 뉴스-이미지 매칭을 위한 완벽한 가이드입니다.

---

## 📋 목차

1. [이미지 리네이밍](#1-이미지-리네이밍)
2. [수동 이미지 매칭](#2-수동-이미지-매칭)
3. [무료 이미지 다운로드](#3-무료-이미지-다운로드)
4. [워크플로우 예제](#4-워크플로우-예제)

---

## 1. 이미지 리네이밍

### 왜 필요한가?

이미지 파일명을 `fire1.jpg`, `fire2.jpg`, ... `fireN.jpg` 형식으로 통일하면:
- ✅ 관리가 쉬워집니다
- ✅ 문서와 스크립트에서 참조가 명확해집니다
- ✅ 새 이미지 추가가 용이합니다
- ✅ 순서가 명확해집니다

### 사용법

#### 기본 사용

```bash
# 1. 미리보기 (실제 변경 없음)
python rename_images.py --input image_data/fire --dry-run

# 2. 실제 리네이밍 (자동으로 백업 폴더 생성됨)
python rename_images.py --input image_data/fire
```

#### 고급 옵션

```bash
# 커스텀 접두사
python rename_images.py --input image_data/fire --prefix disaster

# 시작 번호 변경
python rename_images.py --input image_data/fire --start 10

# PNG로 확장자 변경
python rename_images.py --input image_data/fire --ext .png

# 백업 없이 실행
python rename_images.py --input image_data/fire --no-backup
```

### 실행 결과

```
======================================================================
🖼️  Image Renaming Tool
======================================================================

📁 Found 15 image files in image_data/fire
🏷️  Renaming pattern: fire{N}.jpg (starting from 1)

💾 Backup folder created: image_data/fire/_backup
✅ IMG_20230101_123456.jpg                   → fire1.jpg
✅ photo_fire_scene.jpg                      → fire2.jpg
✅ building_fire.png                         → fire3.jpg
...

======================================================================
✅ Successfully renamed 15 files
💾 Original files backed up to: image_data/fire/_backup
======================================================================
```

### 백업 복원

실수로 잘못 리네이밍한 경우:

```bash
# 백업에서 복원
rm image_data/fire/*.jpg
cp image_data/fire/_backup/* image_data/fire/
```

---

## 2. 수동 이미지 매칭

### 대화형 모드 (추천)

특정 뉴스에 특정 이미지를 수동으로 매칭하는 가장 쉬운 방법입니다.

```bash
python manual_image_matcher.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images image_data/fire
```

### 대화형 모드 사용법

```
============================================================
🎯 Interactive Mapping Mode
============================================================

Commands:
  n [search]   - Show news list (optional: search term)
  i            - Show image list
  m            - Show current mappings
  a            - Add new mapping
  r            - Remove mapping
  s <file>     - Save to file
  q            - Quit
============================================================

> n 대형 화재                    # "대형 화재" 검색
> i                              # 이미지 목록 보기
> a                              # 새 매핑 추가
News doc_id (or index): 1        # 뉴스 선택 (번호 또는 doc_id)
Image filename or index: 3       # 이미지 선택 (번호 또는 파일명)
Caption (optional): 건물 화재 현장  # 캡션 (선택사항)
✅ Added mapping: fire_news_001 → fire3.jpg

> m                              # 현재 매핑 확인
> s manual_mappings.json         # 저장
> q                              # 종료
```

### CLI 모드 (스크립트용)

```bash
# 직접 매칭 추가
python manual_image_matcher.py \
  --add "fire_news_001:fire1.jpg" \
  --add "fire_news_002:fire3.jpg" \
  --add "fire_news_005:fire2.jpg" \
  --output manual_mappings.json

# 기존 매핑 파일 수정
python manual_image_matcher.py \
  --edit manual_mappings.json \
  --add "fire_news_010:fire5.jpg" \
  --remove "fire_news_002" \
  --output manual_mappings_updated.json
```

### 생성되는 매핑 파일 형식

```json
[
  {
    "doc_id": "fire_news_001",
    "image_path": "image_data/fire/fire1.jpg",
    "caption": "건물 화재 현장"
  },
  {
    "doc_id": "fire_news_002",
    "image_path": "image_data/fire/fire3.jpg"
  }
]
```

---

## 3. 무료 이미지 다운로드

### Unsplash API 키 받기

1. https://unsplash.com/developers 접속
2. "Register as a developer" 클릭
3. "New Application" 생성
4. Access Key 복사

### 이미지 다운로드

```bash
# Unsplash에서 다운로드
export UNSPLASH_KEY="your_access_key_here"

python download_free_images.py \
  --source unsplash \
  --api-key $UNSPLASH_KEY \
  --query "fire disaster emergency building" \
  --output image_data/fire_downloaded \
  --limit 30

# 다운로드 후 리네이밍
python rename_images.py --input image_data/fire_downloaded
```

### 검색 쿼리 팁

```bash
# 화재 관련
--query "fire disaster emergency building flames"
--query "firefighter rescue operation"
--query "fire truck emergency response"

# 더 구체적으로
--query "building fire smoke urban city"
--query "fire department emergency"
```

---

## 4. 워크플로우 예제

### 시나리오 A: 처음부터 시작

```bash
# 1. 이미지 다운로드
python download_free_images.py \
  --source unsplash \
  --api-key $UNSPLASH_KEY \
  --query "fire disaster" \
  --output image_data/fire \
  --limit 20

# 2. 이미지 리네이밍
python rename_images.py --input image_data/fire

# 3. 수동 매칭 (대화형)
python manual_image_matcher.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images image_data/fire

# 4. 멀티모달 RAG 생성
python multimodal_rag_v2.py \
  --mode create \
  --collection fire_multimodal \
  --input news_data/01_disaster_Fire_3years.json \
  --images manual_mappings.json
```

### 시나리오 B: 기존 이미지 정리

```bash
# 1. 기존 이미지 리네이밍
python rename_images.py --input image_data/fire --dry-run  # 미리보기
python rename_images.py --input image_data/fire           # 실행

# 2. 수동 매칭
python manual_image_matcher.py \
  --news news_data/01_disaster_Fire_3years.json \
  --images image_data/fire

# 3. RAG 생성
python multimodal_rag_v2.py \
  --mode create \
  --collection fire_multimodal \
  --input news_data/01_disaster_Fire_3years.json \
  --images manual_mappings.json
```

### 시나리오 C: 이미지 추가

```bash
# 1. 새 이미지 다운로드
python download_free_images.py \
  --source unsplash \
  --api-key $UNSPLASH_KEY \
  --query "fire disaster" \
  --output image_data/fire_new \
  --limit 10

# 2. 리네이밍 (기존 번호 이어서)
python rename_images.py \
  --input image_data/fire_new \
  --prefix fire \
  --start 21  # 기존 fire1~fire20이 있다면

# 3. 새 이미지를 기존 폴더로 이동
mv image_data/fire_new/fire*.jpg image_data/fire/

# 4. 기존 매핑에 추가
python manual_image_matcher.py \
  --edit manual_mappings.json \
  --news news_data/01_disaster_Fire_3years.json \
  --images image_data/fire
```

### 시나리오 D: CLI로 빠른 매칭

```bash
# 스크립트나 자동화에 유용
python manual_image_matcher.py \
  --add "fire_news_001:fire1.jpg" \
  --add "fire_news_005:fire2.jpg" \
  --add "fire_news_010:fire3.jpg" \
  --add "fire_news_015:fire4.jpg" \
  --add "fire_news_020:fire5.jpg" \
  --output quick_mappings.json

# RAG 생성
python multimodal_rag_v2.py \
  --mode create \
  --collection fire_quick \
  --input news_data/01_disaster_Fire_3years.json \
  --images quick_mappings.json
```

---

## 💡 팁과 트릭

### 이미지 명명 규칙

- ✅ **권장**: `fire1.jpg`, `fire2.jpg`, ...
- ✅ **카테고리별**: `disaster1.jpg`, `crime1.jpg`, ...
- ✅ **년도별**: `fire2023_1.jpg`, `fire2024_1.jpg`, ...
- ❌ **비권장**: `IMG_20230101.jpg`, `photo_123.jpg`

### 매칭 전략

1. **중요 뉴스 우선**: 주요 뉴스부터 수동 매칭
2. **고품질 이미지 우선**: 좋은 이미지부터 사용
3. **캡션 활용**: 이미지 설명을 캡션으로 저장
4. **정기적 검토**: 주기적으로 매핑 품질 확인

### 백업 관리

```bash
# 백업 폴더 압축
tar -czf image_backup_$(date +%Y%m%d).tar.gz image_data/fire/_backup/

# 오래된 백업 제거
rm -rf image_data/fire/_backup/
```

---

## 🆘 문제 해결

### Q: 리네이밍 후 원본으로 돌아가고 싶어요
A: 백업 폴더에서 복원하세요:
```bash
rm image_data/fire/*.jpg
cp image_data/fire/_backup/* image_data/fire/
```

### Q: 이미지가 중복으로 사용되나요?
A: 네, 한 이미지를 여러 뉴스에 매칭할 수 있습니다. 대화형 모드에서 이미지 목록에 `(2)` 같은 숫자로 사용 횟수가 표시됩니다.

### Q: 매칭을 나중에 수정할 수 있나요?
A: 네, `--edit` 옵션으로 기존 매핑 파일을 수정할 수 있습니다:
```bash
python manual_image_matcher.py --edit manual_mappings.json --news ... --images ...
```

### Q: 새 이미지를 추가하려면?
A: 새 이미지를 `image_data/fire/` 폴더에 추가하고 리네이밍하세요:
```bash
# 기존 fire1~fire20이 있다면
python rename_images.py --input image_data/fire --start 1
# 새 이미지는 자동으로 fire21, fire22, ...로 리네이밍됩니다
```

---

## ✅ 체크리스트

### 이미지 준비
- [ ] 이미지 수집 완료
- [ ] `rename_images.py`로 리네이밍
- [ ] 이미지 품질 확인

### 매칭 작업
- [ ] `manual_image_matcher.py` 대화형 모드 실행
- [ ] 주요 뉴스부터 매칭
- [ ] 캡션 추가 (선택사항)
- [ ] 매핑 파일 저장

### RAG 생성
- [ ] 멀티모달 컬렉션 생성
- [ ] 검색 테스트
- [ ] 결과 확인

---

**이미지 관리를 통해 더 풍부한 멀티모달 RAG를 만드세요! 🚀**

