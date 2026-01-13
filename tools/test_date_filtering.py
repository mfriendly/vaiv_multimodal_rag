#!/usr/bin/env python3
"""
Milvus 날짜 필터링 테스트 스크립트

사용법:
    python test_date_filtering.py --collection fire_multimodal_demo --db-file ./fire_multimodal_demo.db
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from pymilvus import MilvusClient
except ImportError:
    logger.error("pymilvus not installed. Install with: pip install pymilvus")
    exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def test_date_filtering(collection_name: str, db_file: str):
    """날짜 필터링 테스트"""
    
    print("\n" + "="*80)
    print("🧪 Testing Date Range Filtering in Milvus")
    print("="*80)
    
    # Milvus Client 연결
    client = MilvusClient(db_file)
    
    if not client.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' not found in {db_file}")
        return
    
    logger.info(f"✅ Connected to collection: {collection_name}")
    
    # 임베딩 모델 로드
    logger.info("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 테스트 쿼리
    test_query = "화재 사건"
    query_vector = embeddings.embed_query(test_query)
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "No Filter (전체 검색)",
            "filter": None,
            "expected": "모든 날짜의 결과"
        },
        {
            "name": "2021년만 (20210101-20211231)",
            "filter": 'date >= "20210101" && date <= "20211231"',
            "expected": "2021년 데이터만"
        },
        {
            "name": "2022년만 (20220101-20221231)",
            "filter": 'date >= "20220101" && date <= "20221231"',
            "expected": "2022년 데이터만"
        },
        {
            "name": "2023년만 (20230101-20231231)",
            "filter": 'date >= "20230101" && date <= "20231231"',
            "expected": "2023년 데이터만"
        },
        {
            "name": "2022년 여름 (20220601-20220831)",
            "filter": 'date >= "20220601" && date <= "20220831"',
            "expected": "2022년 6-8월 데이터만"
        },
        {
            "name": "Category + Topic 필터",
            "filter": 'category == "disaster" && topic == "fire"',
            "expected": "disaster 카테고리 & fire 토픽만"
        },
        {
            "name": "날짜 + 카테고리 복합 필터",
            "filter": 'date >= "20220101" && date <= "20221231" && category == "disaster"',
            "expected": "2022년 disaster 카테고리만"
        }
    ]
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test {idx}: {test['name']}")
        print(f"Filter: {test['filter'] or 'None'}")
        print(f"Expected: {test['expected']}")
        print(f"{'─'*80}")
        
        try:
            results = client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=5,
                filter=test['filter'],
                output_fields=["doc_id", "title", "date", "category", "topic"]
            )
            
            if results and results[0]:
                print(f"✅ Found {len(results[0])} results:")
                for i, hit in enumerate(results[0], 1):
                    entity = hit.get('entity', {})
                    date = entity.get('date', 'N/A')
                    title = entity.get('title', 'N/A')[:60]
                    category = entity.get('category', 'N/A')
                    topic = entity.get('topic', 'N/A')
                    distance = hit.get('distance', 0)
                    
                    print(f"  [{i}] Date: {date} | Cat: {category} | Topic: {topic}")
                    print(f"      Score: {distance:.4f} | Title: {title}")
            else:
                print("⚠️  No results found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("🎯 Test Summary")
    print("="*80)
    print("날짜 필터링이 제대로 작동하면:")
    print("  - Test 1: 다양한 날짜의 결과가 나와야 함")
    print("  - Test 2-4: 해당 연도의 결과만 나와야 함")
    print("  - Test 5: 2022년 6-8월 결과만 나와야 함")
    print("  - Test 6-7: 필터 조건에 맞는 결과만 나와야 함")
    print("\n✅ 위 결과를 확인하여 날짜 필터링이 제대로 작동하는지 검증하세요!")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test Milvus date filtering')
    parser.add_argument('--collection', '-c', required=True,
                       help='Milvus collection name')
    parser.add_argument('--db-file', default='./fire_multimodal_demo.db',
                       help='Milvus database file')
    
    args = parser.parse_args()
    
    if not Path(args.db_file).exists():
        logger.error(f"Database file not found: {args.db_file}")
        print("\n💡 먼저 DB를 생성하세요:")
        print("   bash demo_fire_multimodal.sh")
        return
    
    test_date_filtering(args.collection, args.db_file)


if __name__ == "__main__":
    main()

