#!/bin/bash

# Multimodal RAG 화재 뉴스 데모 스크립트
# 사용법: bash demo_fire_multimodal.sh

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 설정
NEWS_FILE="news_data/01_disaster_Fire_3years.json"
IMAGE_DIR="image_data/fire"
COLLECTION_NAME="fire_multimodal_demo"
DB_FILE="./multimodal_demo.db"
NUM_NEWS=100
IMAGE_RATIO=0.3

# 출력 파일
PREPARED_NEWS="prepared_fire_news.json"
IMAGE_MAPPINGS="fire_image_mappings.json"

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

# 메인 시작
print_header "🔥 Multimodal RAG Demo - Fire News + Images"

echo ""
echo "This demo will:"
echo "  1. Prepare fire news data with random image assignments"
echo "  2. Create a multimodal Milvus collection"
echo "  3. Insert news + images into the collection"
echo "  4. Demonstrate various search methods:"
echo "     - Text search"
echo "     - Image search"
echo "     - Hybrid search (text + image)"
echo ""
echo "Settings:"
echo "  News file: ${NEWS_FILE}"
echo "  Image dir: ${IMAGE_DIR}"
echo "  Collection: ${COLLECTION_NAME}"
echo "  Database: ${DB_FILE}"
echo "  News limit: ${NUM_NEWS}"
echo "  Image ratio: ${IMAGE_RATIO} (30% of news will have images)"
echo ""

read -p "Press Enter to continue or Ctrl+C to cancel..."

# Step 1: 데이터 준비
print_step "Step 1: Preparing data (assigning images to news)"

if [ ! -f "$NEWS_FILE" ]; then
    print_error "News file not found: $NEWS_FILE"
    print_warning "Please provide a valid news data file"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    print_error "Image directory not found: $IMAGE_DIR"
    print_warning "Please check the image directory path"
    exit 1
fi

python demo_multimodal_fire.py \
    --news "$NEWS_FILE" \
    --images "$IMAGE_DIR" \
    --limit $NUM_NEWS \
    --ratio $IMAGE_RATIO \
    --output-news "$PREPARED_NEWS" \
    --output-images "$IMAGE_MAPPINGS"

if [ $? -eq 0 ]; then
    print_success "Data preparation complete"
else
    print_error "Data preparation failed"
    exit 1
fi

# Step 2: 컬렉션 생성 및 데이터 삽입
print_step "Step 2: Creating multimodal collection and inserting data"

python multimodal_rag_v2.py \
    --mode create \
    --collection "$COLLECTION_NAME" \
    --input "$PREPARED_NEWS" \
    --images "$IMAGE_MAPPINGS" \
    --db-file "$DB_FILE"

if [ $? -eq 0 ]; then
    print_success "Collection created and data inserted"
else
    print_error "Failed to create collection"
    exit 1
fi

# Step 3: 검색 데모
print_header "🔍 Search Demonstrations"

# 3-1: 텍스트 검색
print_step "Demo 1: Text Search - '화재 사건'"
python multimodal_rag_v2.py \
    --mode search \
    --collection "$COLLECTION_NAME" \
    --query "화재 사건" \
    --top-k 5 \
    --db-file "$DB_FILE"

echo ""
read -p "Press Enter to continue to next demo..."

# 3-2: 텍스트 검색 (다른 쿼리)
print_step "Demo 2: Text Search - '대형 화재 진압'"
python multimodal_rag_v2.py \
    --mode search \
    --collection "$COLLECTION_NAME" \
    --query "대형 화재 진압" \
    --top-k 5 \
    --db-file "$DB_FILE"

echo ""
read -p "Press Enter to continue to next demo..."

# 3-3: 이미지 검색
if [ -f "${IMAGE_DIR}/fire1.jpg" ]; then
    print_step "Demo 3: Image Search - using fire1.jpg"
    python multimodal_rag_v2.py \
        --mode search-image \
        --collection "$COLLECTION_NAME" \
        --image "${IMAGE_DIR}/fire1.jpg" \
        --top-k 5 \
        --db-file "$DB_FILE"
    
    echo ""
    read -p "Press Enter to continue to next demo..."
fi

# 3-4: 하이브리드 검색
if [ -f "${IMAGE_DIR}/fire2.jpg" ]; then
    print_step "Demo 4: Hybrid Search - '화재' + fire2.jpg"
    python multimodal_rag_v2.py \
        --mode hybrid \
        --collection "$COLLECTION_NAME" \
        --query "화재" \
        --image "${IMAGE_DIR}/fire2.jpg" \
        --top-k 5 \
        --db-file "$DB_FILE"
    
    echo ""
    read -p "Press Enter to continue..."
fi

# Step 4: 추가 검색 예제
print_header "🎯 Additional Search Examples"

echo ""
echo "You can now run additional searches manually:"
echo ""
echo "1. Text search with different query:"
echo -e "   ${CYAN}python multimodal_rag_v2.py --mode search --collection $COLLECTION_NAME --query '소방관' --db-file $DB_FILE${NC}"
echo ""
echo "2. Image search with different image:"
echo -e "   ${CYAN}python multimodal_rag_v2.py --mode search-image --collection $COLLECTION_NAME --image $IMAGE_DIR/fire3.jpg --db-file $DB_FILE${NC}"
echo ""
echo "3. Hybrid search with custom weights:"
echo -e "   ${CYAN}python multimodal_rag_v2.py --mode hybrid --collection $COLLECTION_NAME --query '건물 화재' --image $IMAGE_DIR/fire1.jpg --db-file $DB_FILE${NC}"
echo ""

# 정리
print_header "📊 Demo Summary"

echo ""
echo "Generated files:"
echo "  - ${PREPARED_NEWS} (prepared news data)"
echo "  - ${IMAGE_MAPPINGS} (image-to-news mappings)"
echo "  - ${DB_FILE} (Milvus database with multimodal collection)"
echo ""
echo "Collection name: ${COLLECTION_NAME}"
echo ""
echo "To clean up (remove all generated files):"
echo -e "  ${CYAN}rm -f ${PREPARED_NEWS} ${IMAGE_MAPPINGS} ${DB_FILE}${NC}"
echo ""

print_success "Demo completed successfully! 🎉"

echo ""
echo "Want to explore more?"
echo "  - Check MULTIMODAL_GUIDE.md for detailed documentation"
echo "  - Try different queries and images"
echo "  - Adjust image assignment ratio"
echo ""

