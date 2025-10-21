#!/usr/bin/env python3
"""
Milvus Multimodal RAG 검색 유틸리티

사용법:
    # 텍스트 검색
    python milvus_rag_search.py --collection fire_news --query "화재 사건" --mode text
    
    # 이미지 검색
    python milvus_rag_search.py --collection fire_news --image path/to/image.jpg --mode image
    
    # 하이브리드 검색 (텍스트 + 이미지)
    python milvus_rag_search.py --collection fire_news --query "화재" --image image.jpg --mode hybrid
"""

import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Milvus imports
from pymilvus import connections, Collection, utility

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# CLIP for image embeddings
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MilvusMultimodalRAG:
    def __init__(
        self,
        collection_name: str,
        milvus_host: str = "localhost",
        milvus_port: str = "19530"
    ):
        """
        Milvus Multimodal RAG 검색 클래스
        
        Args:
            collection_name: 검색할 Milvus 컬렉션 이름
            milvus_host: Milvus 서버 호스트
            milvus_port: Milvus 서버 포트
        """
        self.collection_name = collection_name
        
        # Milvus 연결
        connections.connect(host=milvus_host, port=milvus_port)
        
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        self.collection = Collection(collection_name)
        self.collection.load()
        
        logger.info(f"✅ Connected to collection '{collection_name}' ({self.collection.num_entities} entities)")
        
        # 텍스트 임베딩 모델
        logger.info("Loading text embedding model...")
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda' if self._check_cuda() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 이미지 임베딩 모델 (CLIP)
        self.clip_model = None
        self.clip_processor = None
        if CLIP_AVAILABLE:
            logger.info("Loading CLIP model for image embeddings...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info(f"✅ CLIP model loaded on {self.device}")

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
        텍스트 쿼리로 검색
        
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
        
        # 검색 파라미터
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # 검색 실행
        results = self.collection.search(
            data=[query_embedding],
            anns_field="text_embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        
        # 결과 포맷팅
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "score": hit.score,
                    "id": hit.id,
                }
                result.update(hit.entity.fields)
                formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def search_by_image(
        self,
        image_path: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        이미지로 검색
        
        Args:
            image_path: 이미지 파일 경로
            top_k: 반환할 결과 수
            filter_expr: 필터 표현식
            output_fields: 반환할 필드 리스트
            
        Returns:
            검색 결과 리스트
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            raise ValueError("CLIP model not available. Install transformers and torch.")
        
        logger.info(f"Searching with image: '{image_path}'")
        
        # 이미지 로드 및 임베딩 생성
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        query_embedding = image_features.cpu().numpy().flatten().tolist()
        
        # 기본 출력 필드
        if output_fields is None:
            output_fields = ["doc_id", "title", "content", "date", "category", "topic", "image_url", "image_caption"]
        
        # 검색 파라미터
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # 검색 실행 (이미지 임베딩 사용)
        results = self.collection.search(
            data=[query_embedding],
            anns_field="image_embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        
        # 결과 포맷팅
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "score": hit.score,
                    "id": hit.id,
                }
                result.update(hit.entity.fields)
                formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def hybrid_search(
        self,
        text_query: Optional[str] = None,
        image_path: Optional[str] = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 (텍스트 + 이미지)
        
        Args:
            text_query: 텍스트 쿼리
            image_path: 이미지 파일 경로
            text_weight: 텍스트 검색 가중치
            image_weight: 이미지 검색 가중치
            top_k: 반환할 결과 수
            filter_expr: 필터 표현식
            
        Returns:
            검색 결과 리스트
        """
        logger.info("Performing hybrid search (text + image)")
        
        results_map = {}
        
        # 텍스트 검색
        if text_query:
            text_results = self.search_by_text(text_query, top_k=top_k*2, filter_expr=filter_expr)
            for result in text_results:
                doc_id = result['doc_id']
                if doc_id not in results_map:
                    results_map[doc_id] = {'data': result, 'score': 0}
                results_map[doc_id]['score'] += result['score'] * text_weight
        
        # 이미지 검색
        if image_path and CLIP_AVAILABLE:
            image_results = self.search_by_image(image_path, top_k=top_k*2, filter_expr=filter_expr)
            for result in image_results:
                doc_id = result['doc_id']
                if doc_id not in results_map:
                    results_map[doc_id] = {'data': result, 'score': 0}
                results_map[doc_id]['score'] += result['score'] * image_weight
        
        # 점수 순으로 정렬
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        # 결과 포맷팅
        formatted_results = []
        for item in sorted_results:
            result = item['data']
            result['hybrid_score'] = item['score']
            formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} hybrid results")
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
            print(f"   Score: {result.get('score', result.get('hybrid_score', 0)):.4f}")
            print(f"   Title: {result.get('title', 'N/A')}")
            print(f"   Date: {result.get('date', 'N/A')}")
            print(f"   Category: {result.get('category', 'N/A')} | Topic: {result.get('topic', 'N/A')}")
            print(f"   Source: {result.get('source', 'N/A')}")
            
            if result.get('url'):
                print(f"   URL: {result['url']}")
            
            if result.get('has_image'):
                print(f"   🖼️ Image: {result.get('image_url', 'N/A')}")
            
            if show_content and result.get('content'):
                content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"   Content: {content}")
            
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='Milvus Multimodal RAG Search')
    parser.add_argument('--collection', '-c', required=True,
                       help='Milvus collection name')
    parser.add_argument('--query', '-q',
                       help='Text query')
    parser.add_argument('--image', '-img',
                       help='Image file path')
    parser.add_argument('--mode', '-m', choices=['text', 'image', 'hybrid'], default='text',
                       help='Search mode: text, image, or hybrid')
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
    parser.add_argument('--milvus-host', default='localhost',
                       help='Milvus server host')
    parser.add_argument('--milvus-port', default='19530',
                       help='Milvus server port')
    parser.add_argument('--output', '-o',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # RAG 검색기 초기화
    try:
        rag = MilvusMultimodalRAG(
            collection_name=args.collection,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        return
    
    # 검색 실행
    results = []
    
    try:
        if args.mode == 'text':
            if not args.query:
                logger.error("Text query required for text mode")
                return
            
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
        
        elif args.mode == 'image':
            if not args.image:
                logger.error("Image path required for image mode")
                return
            results = rag.search_by_image(args.image, top_k=args.top_k)
        
        elif args.mode == 'hybrid':
            if not args.query and not args.image:
                logger.error("At least one of query or image required for hybrid mode")
                return
            results = rag.hybrid_search(
                text_query=args.query,
                image_path=args.image,
                top_k=args.top_k
            )
        
        # 결과 출력
        rag.print_results(results, show_content=args.show_content)
        
        # JSON 파일로 저장 (선택적)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Search failed: {e}")


if __name__ == "__main__":
    main()

