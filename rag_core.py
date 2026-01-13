#!/usr/bin/env python3
"""
RAG Core - Multimodal & Text-only RAG Classes

이 모듈은 Milvus 기반 RAG 시스템의 핵심 클래스들을 제공합니다:
- MultimodalRAGV2: 텍스트 + 이미지 멀티모달 RAG
- MilvusLiteRAGV2: 텍스트 전용 RAG (Milvus Lite)

사용법:
    from rag_core import MultimodalRAGV2, MilvusLiteRAGV2
    
    # Multimodal RAG
    rag = MultimodalRAGV2(db_file="./multimodal.db")
    results = rag.search_by_text("화재 사건")
    
    # Text-only RAG
    rag = MilvusLiteRAGV2(collection_name="fire_news", db_file="./fire.db")
    results = rag.search_by_text("화재 사건")
"""

import argparse
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np

# Milvus Client (High-level API)
from pymilvus import MilvusClient

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# CLIP for image embeddings
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
    import requests
    from io import BytesIO
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("transformers not installed. Install with: pip install transformers torch pillow")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalRAGV2:
    def __init__(
        self,
        db_file: str = "./multimodal_rag.db",
        text_model: str = "jhgan/ko-sroberta-multitask",
        clip_model: str = "openai/clip-vit-base-patch32",
    ):
        """
        Multimodal RAG 시스템 (MilvusClient API 사용)
        
        Args:
            db_file: Milvus Lite 데이터베이스 파일
            text_model: 텍스트 임베딩 모델
            clip_model: 이미지 임베딩 모델 (CLIP)
        """
        self.db_file = db_file
        
        # Milvus Client 초기화
        self.client = MilvusClient(db_file)
        logger.info(f"✅ Connected to Milvus at {db_file}")
        
        # 텍스트 임베딩 모델
        logger.info(f"Loading text embedding model: {text_model}")
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name=text_model,
            model_kwargs={'device': 'cuda' if self._check_cuda() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_dim = 768  # ko-sroberta dimension
        
        # 이미지 임베딩 모델 (CLIP)
        self.clip_model = None
        self.clip_processor = None
        self.image_dim = 512  # CLIP dimension
        
        if CLIP_AVAILABLE:
            logger.info(f"Loading CLIP model: {clip_model}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
            logger.info(f"✅ CLIP model loaded on {self.device}")
        else:
            logger.warning("CLIP not available. Image search will be disabled.")

    def _check_cuda(self) -> bool:
        """CUDA 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def create_collection(self, collection_name: str, has_image: bool = True):
        """
        Multimodal 컬렉션 생성
        
        Args:
            collection_name: 컬렉션 이름
            has_image: 이미지 벡터 포함 여부
        """
        # 기존 컬렉션 삭제
        if self.client.has_collection(collection_name):
            logger.info(f"Dropping existing collection '{collection_name}'...")
            self.client.drop_collection(collection_name)
        
        # 새 컬렉션 생성
        # 참고: MilvusClient는 단일 벡터 필드만 지원하므로,
        # 텍스트와 이미지 벡터를 concat하거나 별도 컬렉션 사용
        dimension = self.text_dim + self.image_dim if has_image else self.text_dim
        
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type="COSINE",
            auto_id=True,
            enable_dynamic_field=True,  # 추가 필드 허용
        )
        
        logger.info(f"✅ Created collection '{collection_name}' with dimension={dimension}")

    def load_image(self, image_source: str) -> Optional[Image.Image]:
        """이미지 로드 (로컬 파일 또는 URL)"""
        try:
            if image_source.startswith('http://') or image_source.startswith('https://'):
                response = requests.get(image_source, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_source).convert('RGB')
            return image
        except Exception as e:
            logger.warning(f"Failed to load image from {image_source}: {e}")
            return None

    def encode_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환"""
        return self.text_embeddings.embed_query(text)

    def encode_image(self, image: Image.Image) -> Optional[List[float]]:
        """이미지를 벡터로 변환 (CLIP)"""
        if not CLIP_AVAILABLE or self.clip_model is None:
            return None
        
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def prepare_multimodal_data(
        self,
        news_data: List[Dict[str, Any]],
        image_mappings: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        뉴스 데이터를 Multimodal 형식으로 준비
        
        Args:
            news_data: 뉴스 데이터 리스트
            image_mappings: {doc_id: image_path} 매핑
        
        Returns:
            Milvus 삽입용 데이터
        """
        logger.info(f"Preparing {len(news_data)} items for multimodal storage...")
        
        milvus_data = []
        
        for idx, news in enumerate(tqdm(news_data, desc="Processing items")):
            doc_id = news.get("doc_id", f"doc_{idx}")
            title = news.get("title", "")
            content = news.get("text", news.get("content", ""))
            
            if len(content.strip()) < 50:
                continue
            
            # 텍스트 임베딩
            text_embedding = self.encode_text(content)
            
            # 이미지 임베딩 (있는 경우)
            image_embedding = None
            image_path = ""
            has_image = False
            
            if image_mappings and doc_id in image_mappings:
                image_path = image_mappings[doc_id]
                image = self.load_image(image_path)
                if image:
                    image_embedding = self.encode_image(image)
                    if image_embedding:
                        has_image = True
            
            # 텍스트와 이미지 벡터 결합
            if has_image and image_embedding:
                # Concat: [text_vec, image_vec]
                combined_vector = text_embedding + image_embedding
            else:
                # 이미지 없으면 zero padding
                combined_vector = text_embedding + [0.0] * self.image_dim
            
            # 데이터 항목
            data_item = {
                "vector": combined_vector,
                "doc_id": doc_id,
                "title": title[:500] if title else "",
                "content": content[:5000] if content else "",
                "date": news.get("date", "")[:20],
                "url": news.get("url", "")[:500],
                "source": news.get("source", "")[:200],
                "category": news.get("category", "")[:50],
                "topic": news.get("topic", "")[:100],
                "has_image": has_image,
                "image_path": image_path[:500] if image_path else "",
            }
            
            milvus_data.append(data_item)
        
        logger.info(f"Prepared {len(milvus_data)} items")
        return milvus_data

    def insert_data(self, collection_name: str, data: List[Dict[str, Any]]) -> bool:
        """데이터 삽입"""
        try:
            logger.info(f"Inserting {len(data)} items...")
            
            # 배치 삽입
            batch_size = 1000
            for i in tqdm(range(0, len(data), batch_size), desc="Inserting"):
                batch = data[i:i+batch_size]
                self.client.insert(collection_name=collection_name, data=batch)
            
            logger.info(f"✅ Successfully inserted {len(data)} items")
            return True
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def search_by_text(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """텍스트로 검색"""
        logger.info(f"Searching with text: '{query}'")
        
        # 쿼리 벡터 (텍스트만)
        text_vec = self.encode_text(query)
        query_vec = text_vec + [0.0] * self.image_dim  # Zero padding for image part
        
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vec],
            limit=top_k,
            filter=filter_expr,
            output_fields=["doc_id", "title", "content", "date", "category", "has_image", "image_path"]
        )
        
        return self._format_results(results)

    def search_by_image(
        self,
        collection_name: str,
        image_path: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """이미지로 검색"""
        if not CLIP_AVAILABLE:
            logger.error("CLIP not available for image search")
            return []
        
        logger.info(f"Searching with image: {image_path}")
        
        # 이미지 로드 및 인코딩
        image = self.load_image(image_path)
        if not image:
            return []
        
        image_vec = self.encode_image(image)
        if not image_vec:
            return []
        
        # 쿼리 벡터 (이미지만)
        query_vec = [0.0] * self.text_dim + image_vec  # Zero padding for text part
        
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["doc_id", "title", "content", "has_image", "image_path"]
        )
        
        return self._format_results(results)

    def hybrid_search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_image: Optional[str] = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """하이브리드 검색 (텍스트 + 이미지)"""
        logger.info("Performing hybrid search...")
        
        text_vec = [0.0] * self.text_dim
        image_vec = [0.0] * self.image_dim
        
        # 텍스트 벡터
        if query_text:
            text_vec_raw = self.encode_text(query_text)
            text_vec = [v * text_weight for v in text_vec_raw]
        
        # 이미지 벡터
        if query_image and CLIP_AVAILABLE:
            image = self.load_image(query_image)
            if image:
                image_vec_raw = self.encode_image(image)
                if image_vec_raw:
                    image_vec = [v * image_weight for v in image_vec_raw]
        
        # 결합
        query_vec = text_vec + image_vec
        
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["doc_id", "title", "content", "date", "has_image", "image_path"]
        )
        
        return self._format_results(results)

    def _format_results(self, results) -> List[Dict[str, Any]]:
        """검색 결과 포맷팅"""
        formatted = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.get("id"),
                    "distance": hit.get("distance"),
                }
                result.update(hit.get("entity", {}))
                formatted.append(result)
        return formatted

    def print_results(self, results: List[Dict[str, Any]]):
        """결과 출력"""
        print("\n" + "="*80)
        print(f"🔍 Multimodal Search Results ({len(results)} items)")
        print("="*80)
        
        for idx, result in enumerate(results, 1):
            print(f"\n📄 Result #{idx}")
            print(f"   Distance: {result.get('distance', 0):.4f}")
            print(f"   Title: {result.get('title', 'N/A')}")
            print(f"   Date: {result.get('date', 'N/A')}")
            print(f"   Category: {result.get('category', 'N/A')}")
            
            if result.get('has_image'):
                print(f"   🖼️ Image: {result.get('image_path', 'N/A')}")
            
            content = result.get('content', '')
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"   Content: {preview}")
            
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='Multimodal RAG V2 with MilvusClient')
    parser.add_argument('--mode', required=True,
                       choices=['create', 'search', 'search-image', 'hybrid'],
                       help='Operation mode')
    parser.add_argument('--collection', '-c', required=True,
                       help='Collection name')
    parser.add_argument('--input', '-i',
                       help='Input JSON file (for create mode)')
    parser.add_argument('--images',
                       help='Image mappings JSON file')
    parser.add_argument('--query', '-q',
                       help='Text query (for search modes)')
    parser.add_argument('--image', '-img',
                       help='Image path (for image search)')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                       help='Number of results')
    parser.add_argument('--db-file', default='./multimodal_rag.db',
                       help='Milvus database file')
    
    args = parser.parse_args()
    
    # 초기화
    rag = MultimodalRAGV2(db_file=args.db_file)
    
    if args.mode == 'create':
        # 데이터 로드
        if not args.input:
            logger.error("--input required for create mode")
            return
        
        with open(args.input, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        # 이미지 매핑 로드 (선택적)
        image_mappings = None
        if args.images:
            with open(args.images, 'r', encoding='utf-8') as f:
                image_list = json.load(f)
                image_mappings = {item['doc_id']: item['image_path'] for item in image_list}
        
        # 컬렉션 생성
        rag.create_collection(args.collection, has_image=True)
        
        # 데이터 준비 및 삽입
        data = rag.prepare_multimodal_data(news_data, image_mappings)
        rag.insert_data(args.collection, data)
        
    elif args.mode == 'search':
        if not args.query:
            logger.error("--query required for search mode")
            return
        
        results = rag.search_by_text(args.collection, args.query, top_k=args.top_k)
        rag.print_results(results)
        
    elif args.mode == 'search-image':
        if not args.image:
            logger.error("--image required for search-image mode")
            return
        
        results = rag.search_by_image(args.collection, args.image, top_k=args.top_k)
        rag.print_results(results)
        
    elif args.mode == 'hybrid':
        results = rag.hybrid_search(
            args.collection,
            query_text=args.query,
            query_image=args.image,
            top_k=args.top_k
        )
        rag.print_results(results)


class MilvusLiteRAGV2:
    """Milvus Lite RAG 검색 클래스 (텍스트 전용)"""
    
    def __init__(
        self,
        collection_name: str,
        db_file: str = "./milvus_lite_v2.db",
        text_model: str = "jhgan/ko-sroberta-multitask"
    ):
        """
        Milvus Lite RAG 검색 클래스 V2 (공식 API 사용)
        
        Args:
            collection_name: 검색할 Milvus 컬렉션 이름
            db_file: Milvus Lite 데이터베이스 파일 경로
            text_model: 텍스트 임베딩 모델
        """
        self.collection_name = collection_name
        self.db_file = db_file
        
        # Milvus Client 초기화
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
        logger.info(f"Loading text embedding model: {text_model}")
        device = 'cuda' if self._check_cuda() else 'cpu'
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name=text_model,
            model_kwargs={'device': device},
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
        
        # 검색 실행
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            filter=filter_expr,
            output_fields=output_fields,
        )
        
        # 결과 포맷팅
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.get("id"),
                    "distance": hit.get("distance"),
                    "entity": hit.get("entity", {})
                }
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
        
        if filter_expr:
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


class AnswerGenerator:
    """LLM 기반 답변 생성기 (OpenAI/Anthropic)"""
    
    def __init__(self, llm: str = "openai", model: str = None):
        """
        초기화
        
        Args:
            llm: LLM 제공자 ("openai" 또는 "anthropic")
            model: 모델 이름 (선택사항, 기본값: gpt-4o-mini / claude-sonnet-4-20250514)
        """
        self.llm = llm
        if llm == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model or "gpt-4o-mini"
            logger.info(f"✅ OpenAI client initialized with model: {self.model}")
        else:
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = model or "claude-sonnet-4-20250514"
            logger.info(f"✅ Anthropic client initialized with model: {self.model}")
    
    def generate(self, query: str, results: List[Dict[str, Any]], max_context_chars: int = 8000) -> str:
        """
        검색 결과를 바탕으로 답변 생성
        
        Args:
            query: 사용자 질문
            results: 검색 결과 리스트
            max_context_chars: 최대 컨텍스트 길이
            
        Returns:
            생성된 답변
        """
        context = self._build_context(results, max_context_chars)
        prompt = f"""다음 검색 결과를 바탕으로 질문에 답변하세요. 검색 결과에 없는 내용은 추측하지 마세요.

질문: {query}

검색 결과:
{context}

답변:"""
        
        logger.info(f"Generating answer with {self.llm} ({self.model})")
        
        if self.llm == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        else:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
    
    def _build_context(self, results: List[Dict[str, Any]], max_chars: int) -> str:
        """검색 결과를 컨텍스트 문자열로 변환"""
        chunks = []
        total = 0
        for i, r in enumerate(results):
            entity = r.get('entity', {})
            title = entity.get('title', 'N/A')
            content = entity.get('content', '')[:2000]
            source = entity.get('source', '')
            coll = r.get('_collection', 'unknown')
            chunk = f"[{i+1}] {title}\n출처: {source} (Collection: {coll})\n{content}"
            if total + len(chunk) > max_chars:
                break
            chunks.append(chunk)
            total += len(chunk)
        return "\n\n".join(chunks)


if __name__ == "__main__":
    main()

