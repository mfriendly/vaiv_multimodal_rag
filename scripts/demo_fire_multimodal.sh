#!/bin/bash

# Multimodal RAG 화재 뉴스 데모 스크립트
# ⏰ 시간 범위: 2021년 1월 ~ 2023년 12월 (3년간)
# 🔥 검색 범위: 화재 관련 뉴스 및 이미지만
# 사용법: bash demo_fire_multimodal_improved.sh

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=8
set -e  # 에러 발생 시 중단

# Move to project root (script는 scripts/ 안에 있음)
cd "$(dirname "$0")/.."

# ============================================================
# 데이터 시간 및 주제 범위 제약
# ============================================================
TEMPORAL_SCOPE="2021-01 ~ 2023-12"
TEMPORAL_START="2021년 1월"
TEMPORAL_END="2023년 12월"
CONTENT_SCOPE="화재 관련 뉴스 (Fire-related news only)"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 설정
NEWS_FILE="data/news/01_disaster_Fire_3years.json"
NEWS_IMAGES_DIR="data/images/fire"  # 뉴스에 할당된 실제 이미지
QUERY_IMAGES_DIR="data/query_images/fire"  # 검색용 쿼리 이미지
CLUSTERED_CSV="data/clustered_news.csv"
COLLECTION_NAME="fire_multimodal_demo"
DB_FILE="db/fire_multimodal_demo.db"
NEWS_RANGE="fire_clustered"  # fire_all 또는 fire_clustered

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_step() {
    echo -e "\n${CYAN}▶ $1${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_scope_info() {
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}⏰ 시간 범위: ${TEMPORAL_START} ~ ${TEMPORAL_END}${NC}"
    echo -e "${MAGENTA}🔥 검색 범위: ${CONTENT_SCOPE}${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# 메인 시작
print_header "🔥 Multimodal RAG Demo - Fire News + Images (2021-2023)"

echo ""
print_scope_info
echo ""
echo "This demo will:"
echo "  1. Create multimodal DB from news + images (filename = doc_id)"
echo "  2. Demonstrate various search methods with temporal constraints:"
echo "     - Text search (with time-specific queries)"
echo "     - Image search (fire-related images)"
echo "     - Hybrid search (text + image)"
echo ""
echo "Settings:"
echo "  News file:       ${NEWS_FILE}"
echo "  News images:     ${NEWS_IMAGES_DIR} (doc_id matched)"
echo "  Query images:    ${QUERY_IMAGES_DIR} (for search)"
echo "  News range:      ${NEWS_RANGE}"
if [ "$NEWS_RANGE" = "fire_clustered" ]; then
    echo "  Clustered CSV:   ${CLUSTERED_CSV}"
fi
echo "  Collection:      ${COLLECTION_NAME}"
echo "  Database:        ${DB_FILE}"
echo ""
echo -e "${YELLOW}⚠️  Important Constraints:${NC}"
echo "  - 검색 가능 기간: 2021년 1월 ~ 2023년 12월"
echo "  - 검색 가능 주제: 화재 관련 뉴스만"
echo "  - 쿼리 시 반드시 시간 정보 포함 권장 (예: '2022년 화재', '2023년 여름 화재')"
echo ""

read -p "Press Enter to continue or Ctrl+C to cancel..."

# Step 1: 멀티모달 DB 생성
print_step "Step 1: Creating multimodal DB (auto-matching by filename)"

if [ ! -f "$NEWS_FILE" ]; then
    print_error "News file not found: $NEWS_FILE"
    exit 1
fi

if [ ! -d "$NEWS_IMAGES_DIR" ]; then
    print_error "News images directory not found: $NEWS_IMAGES_DIR"
    print_warning "This directory should contain images named by doc_id (e.g., 202304110010013873784.jpg)"
    exit 1
fi

# DB 생성 명령어 구성
CREATE_CMD="python build_database.py \
    --news \"$NEWS_FILE\" \
    --images \"$NEWS_IMAGES_DIR\" \
    --collection \"$COLLECTION_NAME\" \
    --news-range $NEWS_RANGE \
    --db-file \"$DB_FILE\""

# fire_clustered인 경우 CSV 파일 추가
if [ "$NEWS_RANGE" = "fire_clustered" ]; then
    if [ ! -f "$CLUSTERED_CSV" ]; then
        print_error "Clustered CSV not found: $CLUSTERED_CSV"
        print_warning "Required when news-range=fire_clustered"
        exit 1
    fi
    CREATE_CMD="$CREATE_CMD --clustered-csv \"$CLUSTERED_CSV\""
fi

# DB 생성 실행
eval $CREATE_CMD

if [ $? -eq 0 ]; then
    print_success "Multimodal DB created successfully"
else
    print_error "Failed to create multimodal DB"
    exit 1
fi

# Step 2: 검색 데모
print_header "🔍 Search Demonstrations (Time-Specific Queries)"

# 쿼리 이미지 확인
if [ ! -d "$QUERY_IMAGES_DIR" ]; then
    print_warning "Query images directory not found: $QUERY_IMAGES_DIR"
    print_warning "Image search demos will be skipped"
    QUERY_IMAGES_DIR=""
fi

# 2-1: 텍스트 검색 - 2021년
print_step "Demo 1: Text Search - '2021년 대형 화재 사건'"
echo -e "${YELLOW}Query: 2021년 대형 화재 사건 (시간 범위 명시)${NC}"
python run_search.py \
    --mode text \
    --collection "$COLLECTION_NAME" \
    --query "2021년 대형 화재 사건" \
    --top-k 5 \
    --db-file "$DB_FILE" \
    --date-start "20210101" \
    --date-end "20211231" \
    --category "disaster" \
    --topic "fire"

echo ""
read -p "Press Enter to continue to next demo..."

# 2-2: 텍스트 검색 - 2022년
print_step "Demo 2: Text Search - '2022년 봄 건물 화재'"
echo -e "${YELLOW}Query: 2022년 봄 건물 화재 (시간 + 계절 + 화재 유형)${NC}"
python run_search.py \
    --mode text \
    --collection "$COLLECTION_NAME" \
    --query "2022년 봄 건물 화재" \
    --top-k 5 \
    --db-file "$DB_FILE" \
    --date-start "20220301" \
    --date-end "20220531" \
    --category "disaster" \
    --topic "fire"

echo ""
read -p "Press Enter to continue to next demo..."

# 2-3: 텍스트 검색 - 2023년
print_step "Demo 3: Text Search - '2023년 가을 대형 화재 사건'"
echo -e "${YELLOW}Query: 2023년 가을 대형 화재 사건 (시간 + 계절 + 규모)${NC}"
python run_search.py \
    --mode text \
    --collection "$COLLECTION_NAME" \
    --query "2023년 가을 대형 화재 사건" \
    --top-k 5 \
    --db-file "$DB_FILE" \
    --date-start "20230901" \
    --date-end "20231130" \
    --category "disaster" \
    --topic "fire"

echo ""
read -p "Press Enter to continue to next demo..."

# 2-4: 텍스트 검색 - 특정 월
print_step "Demo 4: Text Search - '2023년 3월 대형 화재 사고'"
echo -e "${YELLOW}Query: 2023년 3월 대형 화재 사고 (연도 + 월 명시)${NC}"
python run_search.py \
    --mode text \
    --collection "$COLLECTION_NAME" \
    --query "2023년 3월 대형 화재 사고" \
    --top-k 5 \
    --db-file "$DB_FILE" \
    --date-start "20230301" \
    --date-end "20230331" \
    --category "disaster" \
    --topic "fire"

echo ""
read -p "Press Enter to continue to next demo..."

# 2-5: 이미지 검색
if [ -n "$QUERY_IMAGES_DIR" ] && [ -f "${QUERY_IMAGES_DIR}/fire1.jpg" ]; then
    print_step "Demo 5: Image Search - using fire1.jpg (화재 진압 현장 이미지)"
    echo -e "${YELLOW}Image Query: fire1.jpg (2021-2023 화재 이미지로 유사 화재 뉴스 검색)${NC}"
    python run_search.py \
        --mode image \
        --collection "$COLLECTION_NAME" \
        --image "${QUERY_IMAGES_DIR}/fire1.jpg" \
        --top-k 5 \
        --db-file "$DB_FILE" \
        --date-start "20210101" \
        --date-end "20231231" \
        --category "disaster" \
        --topic "fire"
    
    echo ""
    read -p "Press Enter to continue to next demo..."
fi

# 2-6: 하이브리드 검색 - 2021년
if [ -n "$QUERY_IMAGES_DIR" ] && [ -f "${QUERY_IMAGES_DIR}/fire2.jpg" ]; then
    print_step "Demo 6: Hybrid Search - '2021년 대형 화재' + fire2.jpg"
    echo -e "${YELLOW}Hybrid Query: Text='2021년 대형 화재' + Image=fire2.jpg${NC}"
    python run_search.py \
        --mode hybrid \
        --collection "$COLLECTION_NAME" \
        --query "2021년 대형 화재" \
        --image "${QUERY_IMAGES_DIR}/fire2.jpg" \
        --top-k 5 \
        --db-file "$DB_FILE" \
        --date-start "20210101" \
        --date-end "20211231" \
        --category "disaster" \
        --topic "fire"
    
    echo ""
    read -p "Press Enter to continue to next demo..."
fi

# 2-7: 하이브리드 검색 - 2022년
if [ -n "$QUERY_IMAGES_DIR" ] && [ -f "${QUERY_IMAGES_DIR}/fire3.jpg" ]; then
    print_step "Demo 7: Hybrid Search - '2022년 여름 대형 화재 사고' + fire3.jpg"
    echo -e "${YELLOW}Hybrid Query: Text='2022년 여름 대형 화재 사고' + Image=fire3.jpg${NC}"
    python run_search.py \
        --mode hybrid \
        --collection "$COLLECTION_NAME" \
        --query "2022년 여름 대형 화재 사고" \
        --image "${QUERY_IMAGES_DIR}/fire3.jpg" \
        --top-k 5 \
        --db-file "$DB_FILE" \
        --date-start "20220601" \
        --date-end "20220831" \
        --category "disaster" \
        --topic "fire"
    
    echo ""
    read -p "Press Enter to continue..."
fi

# Step 3: 추가 검색 예제
print_header "🎯 Additional Search Examples (Time-Specific)"

echo ""
echo "권장 검색 쿼리 형식 (시간 정보 포함):"
echo ""
echo "1. 연도 기반 검색 (날짜 필터 적용):"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2021년 화재 사건' --db-file $DB_FILE --date-start 20210101 --date-end 20211231${NC}"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2022년 공장 화재' --db-file $DB_FILE --date-start 20220101 --date-end 20221231${NC}"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2023년 주택 화재' --db-file $DB_FILE --date-start 20230101 --date-end 20231231${NC}"
echo ""

echo "2. 연도 + 계절 검색 (정확한 날짜 범위):"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2021년 봄 화재' --db-file $DB_FILE --date-start 20210301 --date-end 20210531${NC}"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2022년 여름 산불' --db-file $DB_FILE --date-start 20220601 --date-end 20220831${NC}"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2023년 겨울 화재 사고' --db-file $DB_FILE --date-start 20231201 --date-end 20231231${NC}"
echo ""

echo "3. 연도 + 월 검색 (더 구체적):"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2021년 1월 화재' --db-file $DB_FILE --date-start 20210101 --date-end 20210131${NC}"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2022년 6월 대형 화재' --db-file $DB_FILE --date-start 20220601 --date-end 20220630${NC}"
echo -e "   ${CYAN}python run_search.py --mode text --collection $COLLECTION_NAME --query '2023년 10월 건물 화재' --db-file $DB_FILE --date-start 20231001 --date-end 20231031${NC}"
echo ""

if [ -n "$QUERY_IMAGES_DIR" ]; then
    echo "4. 이미지 검색 (날짜 범위 + 화재 필터):"
    echo -e "   ${CYAN}python run_search.py --mode image --collection $COLLECTION_NAME --image $QUERY_IMAGES_DIR/fire1.jpg --db-file $DB_FILE --date-start 20210101 --date-end 20231231 --topic fire${NC}"
    echo -e "   ${CYAN}python run_search.py --mode image --collection $COLLECTION_NAME --image $QUERY_IMAGES_DIR/fire4.jpg --db-file $DB_FILE --date-start 20220101 --date-end 20221231 --category disaster${NC}"
    echo ""
    echo "5. 하이브리드 검색 (시간 + 이미지 + 필터):"
    echo -e "   ${CYAN}python run_search.py --mode hybrid --collection $COLLECTION_NAME --query '2022년 화재' --image $QUERY_IMAGES_DIR/fire5.jpg --db-file $DB_FILE --date-start 20220101 --date-end 20221231${NC}"
    echo -e "   ${CYAN}python run_search.py --mode hybrid --collection $COLLECTION_NAME --query '2023년 3월 건물 화재' --image $QUERY_IMAGES_DIR/fire6.jpg --db-file $DB_FILE --date-start 20230301 --date-end 20230331${NC}"
    echo ""
fi

echo ""
echo -e "${YELLOW}⚠️  검색 제약 사항 및 필터링:${NC}"
echo "  - 시간 범위: 2021년 1월 ~ 2023년 12월만 검색 가능"
echo "  - 검색 주제: 화재 관련 키워드만 유효"
echo "  - 권장 키워드: 화재, 불, 산불, 건물화재, 공장화재, 주택화재, 화재사고, 소방 등"
echo ""
echo -e "${GREEN}✅ 날짜 필터링 적용됨:${NC}"
echo "  - --date-start: 시작 날짜 (YYYYMMDD 형식)"
echo "  - --date-end: 종료 날짜 (YYYYMMDD 형식)"
echo "  - --category: 카테고리 필터 (예: disaster)"
echo "  - --topic: 토픽 필터 (예: fire)"
echo "  - Milvus 필터 표현식으로 DB 레벨에서 필터링됨 (FAISS처럼 작동)"
echo ""

# 정리
print_header "📊 Demo Summary"

echo ""
print_scope_info
echo ""
echo "Generated files:"
echo "  - ${DB_FILE} (Milvus database with multimodal collection)"
echo ""
echo "Collection name: ${COLLECTION_NAME}"
echo "News range: ${NEWS_RANGE}"
echo "Temporal scope: ${TEMPORAL_SCOPE}"
echo "Content scope: Fire-related news only"
echo ""
echo "To clean up (remove database file):"
echo -e "  ${CYAN}rm -f ${DB_FILE}*${NC}"
echo ""

print_success "Demo completed successfully! 🎉"

echo ""
echo "검색 팁:"
echo "  ✅ DO: '2021년 화재' + --date-start 20210101 --date-end 20211231"
echo "  ✅ DO: '2022년 3월 건물 화재' + --date-start 20220301 --date-end 20220331"
echo "  ✅ DO: '2023년 여름 산불' + --date-start 20230601 --date-end 20230831"
echo "  ❌ DON'T: '2020년 화재' (범위 밖), '2024년 화재' (범위 밖), '홍수' (화재 아님)"
echo ""
echo "날짜 필터링 작동 방식:"
echo "  - Milvus 필터 표현식: date >= \"YYYYMMDD\" && date <= \"YYYYMMDD\""
echo "  - DB 레벨에서 필터링되므로 FAISS의 메타데이터 필터링과 동일한 효과"
echo "  - category 및 topic 필터도 동시 적용 가능"
echo ""
echo "더 알아보기:"
echo "  - 자세한 문서는 README.md 참조"
echo "  - 다양한 화재 관련 쿼리 이미지는 ${QUERY_IMAGES_DIR} 확인"
echo "  - NEWS_RANGE를 'fire_all'로 변경하면 전체 뉴스 사용 가능"
echo ""