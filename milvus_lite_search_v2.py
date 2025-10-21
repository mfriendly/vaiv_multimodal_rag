#!/usr/bin/env python3
"""
Milvus Lite 검색 유틸리티 V2 (공식 API 사용)

사용법:
    python milvus_lite_search_v2.py --collection fire_news --query "화재 사건"
    python milvus_lite_search_v2.py --collection fire_news --query "화재" --category disaster
"""

import argparse
import logging
from typing import List, Dict, Any, Optional
import json

# Milvus Client (High-level API)
from pymilvus import MilvusClient

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MilvusLiteRAGV2:
    def __init__(
        self,
        collection_name: str,
        db_file: str = "./milvus_lite_v2.db"
    ):
        """
        Milvus Lite RAG 검색 클래스 V2 (공식 API 사용)
        
        Args:
            collection_name: 검색할 Milvus 컬렉션 이름
            db_file: Milvus Lite 데이터베이스 파일 경로
        """
        self.collection_name = collection_name
        self.db_file = db_file
        
        # Milvus Client 초기화 (High-level API!)
        self.client = MilvusClient(db_file)
        
        # 컬렉션 존재 확인
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist in {db_file}")
        
        # 컬렉션 통계
        stats = self.client.get_collection_stats(collection_name)
        logger.info(f"✅ Connected to collection '{collection_name}'")
        logger.info(f"📊 Collection stats: {stats}")
        logger.info(f"📁 Database: {db_file}")
        
        # 텍스트 임베딩 모델
        logger.info("Loading text embedding model...")
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda' if self._check_cuda() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _check_cuda(self) -> bool:
        """CUDA 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        텍스트 쿼리로 검색 (MilvusClient API 사용)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_expr: 필터 표현식 (예: 'category == "disaster"')
            output_fields: 반환할 필드 리스트
            
        Returns:
            검색 결과 리스트
        """
        logger.info(f"Searching with text query: '{query}'")
        
        # 쿼리 임베딩 생성
        query_embedding = self.text_embeddings.embed_query(query)
        
        # 기본 출력 필드
        if output_fields is None:
            output_fields = ["doc_id", "title", "content", "date", "category", "topic", "url", "source"]
        
        # 검색 실행 (MilvusClient.search()!)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],  # 쿼리 벡터 리스트
            limit=top_k,
            filter=filter_expr,  # 'filter' 파라미터 (not 'expr')
            output_fields=output_fields,
            # search_params는 자동으로 설정됨
        )
        
        # 결과 포맷팅
        formatted_results = []
        for hits in results:  # results는 쿼리별 결과 리스트
            for hit in hits:  # hits는 단일 쿼리의 결과들
                result = {
                    "id": hit.get("id"),
                    "distance": hit.get("distance"),  # 거리 (작을수록 유사)
                    "entity": hit.get("entity", {})  # 엔티티 필드
                }
                # 엔티티 필드를 최상위로 이동
                result.update(result["entity"])
                formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def search_with_metadata_filter(
        self,
        query: str,
        category: Optional[str] = None,
        topic: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        메타데이터 필터를 사용한 검색
        
        Args:
            query: 검색 쿼리
            category: 카테고리 필터
            topic: 토픽 필터
            date_start: 시작 날짜 (YYYYMMDD)
            date_end: 종료 날짜 (YYYYMMDD)
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        # 필터 표현식 구성
        filter_parts = []
        
        if category:
            filter_parts.append(f'category == "{category}"')
        
        if topic:
            filter_parts.append(f'topic == "{topic}"')
        
        if date_start:
            filter_parts.append(f'date >= "{date_start}"')
        
        if date_end:
            filter_parts.append(f'date <= "{date_end}"')
        
        filter_expr = " && ".join(filter_parts) if filter_parts else None
        
        logger.info(f"Searching with filter: {filter_expr}")
        
        return self.search_by_text(query, top_k=top_k, filter_expr=filter_expr)

    def print_results(self, results: List[Dict[str, Any]], show_content: bool = False):
        """검색 결과를 보기 좋게 출력"""
        print("\n" + "="*80)
        print(f"🔍 Search Results ({len(results)} items)")
        print("="*80)
        
        for idx, result in enumerate(results, 1):
            print(f"\n📄 Result #{idx}")
            print(f"   Distance: {result.get('distance', 0):.4f}")
            print(f"   ID: {result.get('id', 'N/A')}")
            print(f"   Title: {result.get('title', 'N/A')}")
            print(f"   Date: {result.get('date', 'N/A')}")
            print(f"   Category: {result.get('category', 'N/A')} | Topic: {result.get('topic', 'N/A')}")
            print(f"   Source: {result.get('source', 'N/A')}")
            
            if result.get('url'):
                print(f"   URL: {result['url']}")
            
            if show_content and result.get('content'):
                content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"   Content: {content}")
            
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='Milvus Lite Search V2 (Official API)')
    parser.add_argument('--collection', '-c', required=True,
                       help='Milvus collection name')
    parser.add_argument('--query', '-q', required=True,
                       help='Text query')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                       help='Number of results to return')
    parser.add_argument('--category',
                       help='Filter by category')
    parser.add_argument('--topic',
                       help='Filter by topic')
    parser.add_argument('--date-start',
                       help='Filter by start date (YYYYMMDD)')
    parser.add_argument('--date-end',
                       help='Filter by end date (YYYYMMDD)')
    parser.add_argument('--show-content', action='store_true',
                       help='Show content in results')
    parser.add_argument('--db-file', default='./milvus_lite_v2.db',
                       help='Milvus Lite database file path')
    parser.add_argument('--output', '-o',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # RAG 검색기 초기화
    try:
        rag = MilvusLiteRAGV2(
            collection_name=args.collection,
            db_file=args.db_file
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 검색 실행
    try:
        # 메타데이터 필터가 있으면 사용
        if args.category or args.topic or args.date_start or args.date_end:
            results = rag.search_with_metadata_filter(
                query=args.query,
                category=args.category,
                topic=args.topic,
                date_start=args.date_start,
                date_end=args.date_end,
                top_k=args.top_k
            )
        else:
            results = rag.search_by_text(args.query, top_k=args.top_k)
        
        # 결과 출력
        rag.print_results(results, show_content=args.show_content)
        
        # JSON 파일로 저장 (선택적)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

