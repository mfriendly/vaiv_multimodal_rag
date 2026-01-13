#!/bin/bash

# Multimodal RAG 화재 뉴스 데모 - 동영상 촬영용
# DB 생성은 건너뛰고 검색 데모만 진행
# 사용법: bash demo_fire_video.sh [demo_number]
#   demo_number: 1, 2, 3 (생략시 전체 진행)

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=8
set -e

# Move to project root (script는 scripts/ 안에 있음)
cd "$(dirname "$0")/.."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 설정
COLLECTION_NAME="fire_multimodal_demo"
DB_FILE="db/fire_multimodal_demo.db"
QUERY_IMAGES_DIR="data/query_images/fire"

# 유틸리티 함수
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

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

# DB 확인
if [ ! -f "$DB_FILE" ]; then
    echo -e "${RED}❌ Database not found: $DB_FILE${NC}"
    echo ""
    echo "Please create the database first by running:"
    echo "  bash demo_fire_multimodal.sh"
    echo ""
    exit 1
fi

# =============================================================================
# 동영상 1: 텍스트 검색 데모 (30초)
# =============================================================================
demo1_text_search() {
    clear
    print_header "🎬 동영상 1: 텍스트 검색 데모 (날짜 필터링)"
    
    echo ""
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}⏰ 시간 범위: 2021년 1월 ~ 2023년 12월${NC}"
    echo -e "${MAGENTA}🔥 검색 범위: 화재 관련 뉴스 (Fire-related news only)${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    sleep 2
    
    # 2021년 검색
    print_step "Demo 1-1: Text Search - '2021년 화재 사건'"
    print_info "Query: 2021년 화재 사건 (시간 범위: 2021.01.01-2021.12.31)"
    echo ""
    
    python run_search.py \
        --mode text \
        --collection "$COLLECTION_NAME" \
        --query "2021년 화재 사건" \
        --top-k 3 \
        --db-file "$DB_FILE" \
        --date-start "20210101" \
        --date-end "20211231" \
        --category "disaster" \
        --topic "fire" \
        --non-interactive
    
    echo ""
    sleep 3
    
    # 2022년 검색
    print_step "Demo 1-2: Text Search - '2022년 건물 화재'"
    print_info "Query: 2022년 건물 화재 (시간 범위: 2022.01.01-2022.12.31)"
    echo ""
    
    python run_search.py \
        --mode text \
        --collection "$COLLECTION_NAME" \
        --query "2022년 건물 화재" \
        --top-k 3 \
        --db-file "$DB_FILE" \
        --date-start "20220101" \
        --date-end "20221231" \
        --category "disaster" \
        --topic "fire" \
        --non-interactive
    
    echo ""
    sleep 2
    
    print_success "✅ 텍스트 검색 데모 완료!"
    echo ""
}

# =============================================================================
# 동영상 2: 이미지 & 하이브리드 검색 데모 (40초)
# =============================================================================
demo2_multimodal_search() {
    clear
    print_header "🎬 동영상 2: 이미지 & 하이브리드 검색 데모"
    
    echo ""
    echo -e "${CYAN}💡 멀티모달 검색: 이미지 기반 + 텍스트+이미지 결합${NC}"
    echo ""
    
    sleep 2
    
    # 이미지 검색
    if [ -f "${QUERY_IMAGES_DIR}/fire1.jpg" ]; then
        print_step "Demo 2-1: Image Search - fire1.jpg"
        print_info "Image Query: 화재 현장 이미지로 유사 뉴스 검색"
        echo ""
        
        python run_search.py \
            --mode image \
            --collection "$COLLECTION_NAME" \
            --image "${QUERY_IMAGES_DIR}/fire1.jpg" \
            --top-k 3 \
            --db-file "$DB_FILE" \
            --date-start "20210101" \
            --date-end "20231231" \
            --category "disaster" \
            --topic "fire" \
            --non-interactive
        
        echo ""
        sleep 3
    else
        echo -e "${RED}⚠️  쿼리 이미지를 찾을 수 없습니다: ${QUERY_IMAGES_DIR}/fire1.jpg${NC}"
    fi
    
    # 하이브리드 검색
    if [ -f "${QUERY_IMAGES_DIR}/fire2.jpg" ]; then
        print_step "Demo 2-2: Hybrid Search - '2021년 화재' + fire2.jpg"
        print_info "Hybrid Query: 텍스트와 이미지를 결합한 검색"
        echo ""
        echo -e "${CYAN}💡 Late Fusion: 화재 유형 → 이미지 가중치 60% 적용${NC}"
        echo ""
        
        python run_search.py \
            --mode hybrid \
            --collection "$COLLECTION_NAME" \
            --query "2021년 화재" \
            --image "${QUERY_IMAGES_DIR}/fire2.jpg" \
            --top-k 3 \
            --db-file "$DB_FILE" \
            --date-start "20210101" \
            --date-end "20211231" \
            --category "disaster" \
            --topic "fire" \
            --non-interactive
        
        echo ""
        sleep 2
    else
        echo -e "${RED}⚠️  쿼리 이미지를 찾을 수 없습니다: ${QUERY_IMAGES_DIR}/fire2.jpg${NC}"
    fi
    
    print_success "✅ 멀티모달 검색 데모 완료!"
    echo ""
}

# =============================================================================
# 동영상 3: 필터링 효과 & 인터랙티브 기능 (30초)
# =============================================================================
demo3_filtering_interactive() {
    clear
    print_header "🎬 동영상 3: 날짜 필터링 효과 & 이미지 링크"
    
    echo ""
    echo -e "${CYAN}💡 Milvus 메타데이터 필터링의 성능 향상${NC}"
    echo ""
    
    sleep 2
    
    # 필터 없는 검색 (비교용)
    print_step "비교 1: 필터 없는 검색 (전체 기간)"
    print_info "Query: 화재 (필터 없음)"
    echo ""
    
    echo -e "${YELLOW}⏱️  검색 시간 측정 중...${NC}"
    time_output=$(TIMEFORMAT='%3R'; { time python run_search.py \
        --mode text \
        --collection "$COLLECTION_NAME" \
        --query "화재" \
        --top-k 3 \
        --db-file "$DB_FILE" \
        --non-interactive 2>&1; } 2>&1 | tail -n 20)
    
    echo "$time_output" | head -n 15
    echo ""
    sleep 2
    
    # 필터 있는 검색
    print_step "비교 2: 날짜 필터 적용 (2021년만)"
    print_info "Query: 화재 (2021년 1월~12월 필터 적용)"
    echo ""
    
    echo -e "${YELLOW}⏱️  검색 시간 측정 중...${NC}"
    time_output=$(TIMEFORMAT='%3R'; { time python run_search.py \
        --mode text \
        --collection "$COLLECTION_NAME" \
        --query "화재" \
        --top-k 3 \
        --db-file "$DB_FILE" \
        --date-start "20210101" \
        --date-end "20211231" \
        --category "disaster" \
        --topic "fire" \
        --non-interactive 2>&1; } 2>&1 | tail -n 20)
    
    echo "$time_output" | head -n 15
    echo ""
    
    echo -e "${GREEN}⚡ 필터링 효과: DB 레벨에서 빠른 검색!${NC}"
    echo ""
    sleep 3
    
    # 인터랙티브 기능 시뮬레이션
    print_step "클릭 가능한 이미지 링크 데모"
    echo ""
    
    # 검색 실행 (결과 3개만)
    python run_search.py \
        --mode text \
        --collection "$COLLECTION_NAME" \
        --query "화재 사건" \
        --top-k 3 \
        --db-file "$DB_FILE" \
        --date-start "20210101" \
        --date-end "20211231" \
        --non-interactive 2>&1 | head -n 40
    
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}💡 인터랙티브 모드 (--non-interactive 없이 실행 시):${NC}"
    echo "   • 전체 내용 보기: 결과 번호 입력 (예: 1, 2, 3)"
    echo "   • 이미지 링크 보기: i+번호 입력 (예: i1, i2)"
    echo "     → 클릭 가능한 파일 링크가 표시됩니다"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
    echo ""
    
    sleep 3
    
    print_success "✅ 필터링 & 인터랙티브 데모 완료!"
    echo ""
}

# =============================================================================
# 메인 실행 로직
# =============================================================================

show_menu() {
    clear
    print_header "🎥 MMRRM 동영상 촬영 메뉴"
    echo ""
    echo "동영상 촬영용 데모 스크립트"
    echo "  Database: $DB_FILE"
    echo "  Collection: $COLLECTION_NAME"
    echo ""
    echo "촬영 시나리오:"
    echo ""
    echo "  1. 텍스트 검색 데모 (30초)"
    echo "     - 2021년, 2022년 텍스트 검색"
    echo "     - 날짜 필터링 적용"
    echo ""
    echo "  2. 멀티모달 검색 데모 (40초)"
    echo "     - 이미지 검색"
    echo "     - 하이브리드 검색 (텍스트+이미지)"
    echo "     - Late Fusion 가중치 설명"
    echo ""
    echo "  3. 필터링 & 인터랙티브 (30초)"
    echo "     - 필터 전후 비교"
    echo "     - 클릭 가능한 이미지 링크"
    echo ""
    echo "  all. 전체 데모 연속 실행"
    echo ""
    echo "  q. 종료"
    echo ""
    echo -e "${YELLOW}📹 녹화 소프트웨어를 먼저 시작하세요!${NC}"
    echo ""
}

# 인자가 있으면 직접 실행
if [ $# -eq 1 ]; then
    case "$1" in
        1)
            demo1_text_search
            ;;
        2)
            demo2_multimodal_search
            ;;
        3)
            demo3_filtering_interactive
            ;;
        all)
            echo -e "${YELLOW}📹 전체 데모를 3초 후 시작합니다...${NC}"
            sleep 3
            demo1_text_search
            echo ""
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${CYAN}  3초 후 다음 데모로 진행...${NC}"
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            sleep 3
            demo2_multimodal_search
            echo ""
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${CYAN}  3초 후 마지막 데모로 진행...${NC}"
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            sleep 3
            demo3_filtering_interactive
            ;;
        *)
            echo -e "${RED}잘못된 옵션: $1${NC}"
            echo "사용법: bash demo_fire_video.sh [1|2|3|all]"
            exit 1
            ;;
    esac
    exit 0
fi

# 인자가 없으면 메뉴 표시
while true; do
    show_menu
    read -p "선택 (1/2/3/all/q): " choice
    
    case "$choice" in
        1)
            echo ""
            echo -e "${YELLOW}📹 동영상 1 촬영을 3초 후 시작합니다...${NC}"
            sleep 3
            demo1_text_search
            read -p "Press Enter to return to menu..."
            ;;
        2)
            echo ""
            echo -e "${YELLOW}📹 동영상 2 촬영을 3초 후 시작합니다...${NC}"
            sleep 3
            demo2_multimodal_search
            read -p "Press Enter to return to menu..."
            ;;
        3)
            echo ""
            echo -e "${YELLOW}📹 동영상 3 촬영을 3초 후 시작합니다...${NC}"
            sleep 3
            demo3_filtering_interactive
            read -p "Press Enter to return to menu..."
            ;;
        all)
            echo ""
            echo -e "${YELLOW}📹 전체 데모를 3초 후 시작합니다...${NC}"
            sleep 3
            demo1_text_search
            sleep 3
            demo2_multimodal_search
            sleep 3
            demo3_filtering_interactive
            read -p "Press Enter to return to menu..."
            ;;
        q|Q)
            echo ""
            echo -e "${GREEN}👋 종료합니다.${NC}"
            exit 0
            ;;
        *)
            echo ""
            echo -e "${RED}잘못된 선택입니다.${NC}"
            sleep 2
            ;;
    esac
done

