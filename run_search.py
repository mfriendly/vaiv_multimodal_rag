#!/usr/bin/env python3
"""
Multimodal RAG Search - 텍스트/이미지/하이브리드 검색

검색 모드:
  - text: 텍스트 검색
  - image: 이미지 검색
  - hybrid: 텍스트 + 이미지 하이브리드 검색

다중 컬렉션 검색:
  --collection fire_news,disaster_manual  # 쉼표로 구분하여 여러 컬렉션 동시 검색
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Milvus Client
from pymilvus import MilvusClient

# Answer Generator
from rag_core import AnswerGenerator

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
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class MultimodalSearcher:
    """멀티모달 검색 클래스 (다중 컬렉션 지원)"""
    
    def __init__(
        self,
        db_file: str,
        collections: List[str],
        text_model: str = "jhgan/ko-sroberta-multitask",
        clip_model: str = "openai/clip-vit-base-patch32"
    ):
        """초기화 - collections는 리스트 또는 단일 문자열"""
        self.collections = collections if isinstance(collections, list) else [collections]
        
        # Milvus Client
        self.client = MilvusClient(db_file)
        logger.info(f"✅ Connected to {db_file}, collections: {self.collections}")
        
        # Text embedding
        device = 'cuda' if self._check_cuda() else 'cpu'
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name=text_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"✅ Text model loaded on {device}")
        
        # CLIP
        if CLIP_AVAILABLE:
            self.device = "cuda" if self._check_cuda() else "cpu"
            self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
            logger.info(f"✅ CLIP model loaded on {self.device}")
    
    def _check_cuda(self) -> bool:
        """CUDA 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def encode_text(self, text: str) -> List[float]:
        """텍스트 임베딩"""
        return self.text_embeddings.embed_query(text)
    
    def encode_image(self, image_path: str) -> List[float]:
        """이미지 임베딩"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None
    
    def search_by_text(self, query: str, top_k: int = 5, 
                      date_start: str = None, date_end: str = None,
                      category: str = None, topic: str = None) -> List[Dict[str, Any]]:
        """텍스트 검색 (다중 컬렉션, 날짜 범위 필터링 지원)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            date_start: 시작 날짜 (YYYYMMDD 형식)
            date_end: 종료 날짜 (YYYYMMDD 형식)
            category: 카테고리 필터
            topic: 토픽 필터
        """
        logger.info(f"Searching by text: {query} in collections: {self.collections}")
        
        # 필터 표현식 구성
        filter_parts = []
        if date_start:
            filter_parts.append(f'date >= "{date_start}"')
        if date_end:
            filter_parts.append(f'date <= "{date_end}"')
        if category:
            filter_parts.append(f'category == "{category}"')
        if topic:
            filter_parts.append(f'topic == "{topic}"')
        
        filter_expr = " && ".join(filter_parts) if filter_parts else None
        
        if filter_expr:
            logger.info(f"Applying filter: {filter_expr}")
        
        query_vector = self.encode_text(query)
        all_results = []
        
        # 모든 컬렉션에서 검색
        for coll in self.collections:
            if not self.client.has_collection(coll):
                logger.warning(f"Collection '{coll}' not found, skipping")
                continue
            
            results = self.client.search(
                collection_name=coll,
                data=[query_vector],
                limit=top_k,
                filter=filter_expr,
                output_fields=["doc_id", "title", "content", "date", "source", "has_image", "image_path", "category", "topic"]
            )
            
            for hit in (results[0] if results else []):
                hit['_collection'] = coll  # 어느 컬렉션에서 왔는지 표시
                all_results.append(hit)
        
        # 스코어로 정렬 후 top_k 반환
        all_results.sort(key=lambda x: x.get('distance', 0), reverse=True)
        return all_results[:top_k]
    
    def search_by_image(self, image_path: str, top_k: int = 5,
                       date_start: str = None, date_end: str = None,
                       category: str = None, topic: str = None) -> List[Dict[str, Any]]:
        """이미지 검색 (다중 컬렉션, image_embedding JSON 필드 기반)
        
        Args:
            image_path: 쿼리 이미지 경로
            top_k: 반환할 결과 수
            date_start: 시작 날짜 (YYYYMMDD 형식)
            date_end: 종료 날짜 (YYYYMMDD 형식)
            category: 카테고리 필터
            topic: 토픽 필터
        """
        if not CLIP_AVAILABLE:
            logger.error("CLIP not available for image search")
            return []
        
        logger.info(f"Searching by image: {image_path} in collections: {self.collections}")
        
        # 쿼리 이미지 임베딩
        query_image_vector = self.encode_image(image_path)
        if query_image_vector is None:
            return []
        
        # 필터 표현식 구성
        filter_parts = ["has_image == true"]  # 이미지가 있는 문서만
        if date_start:
            filter_parts.append(f'date >= "{date_start}"')
        if date_end:
            filter_parts.append(f'date <= "{date_end}"')
        if category:
            filter_parts.append(f'category == "{category}"')
        if topic:
            filter_parts.append(f'topic == "{topic}"')
        
        filter_expr = " && ".join(filter_parts)
        
        if filter_expr:
            logger.info(f"Applying filter: {filter_expr}")
        
        import numpy as np
        all_scored_results = []
        
        # 모든 컬렉션에서 검색
        for coll in self.collections:
            if not self.client.has_collection(coll):
                logger.warning(f"Collection '{coll}' not found, skipping")
                continue
            
            logger.info(f"Searching images in collection: {coll}")
            
            results = self.client.search(
                collection_name=coll,
                data=[self.encode_text("화재")],  # 일반적인 쿼리
                limit=1000,
                output_fields=["doc_id", "title", "content", "date", "source", "has_image", "image_path", "image_embedding", "category", "topic"],
                filter=filter_expr
            )
            
            for result in (results[0] if results else []):
                entity = result.get('entity', {})
                
                img_emb_str = entity.get('image_embedding', '[]')
                if not img_emb_str or img_emb_str == '[]':
                    continue
                
                stored_image_vector = json.loads(img_emb_str)
                query_norm = np.linalg.norm(query_image_vector)
                stored_norm = np.linalg.norm(stored_image_vector)
                
                if query_norm > 0 and stored_norm > 0:
                    similarity = np.dot(query_image_vector, stored_image_vector) / (query_norm * stored_norm)
                    all_scored_results.append({
                        'entity': entity,
                        'distance': float(similarity),
                        '_collection': coll
                    })
        
        # 유사도 기준으로 정렬 (높은 순)
        all_scored_results.sort(key=lambda x: x['distance'], reverse=True)
        return all_scored_results[:top_k]
    
    def print_results(self, results: List[Dict[str, Any]], mode: str, interactive: bool = True):
        """검색 결과 출력"""
        print(f"\n{'='*80}")
        print(f"🔍 {mode.upper()} Search Results")
        print(f"{'='*80}\n")
        
        if not results:
            print("No results found.")
            return
        
        for idx, result in enumerate(results, 1):
            entity = result.get('entity', {})
            distance = result.get('distance', 0)
            coll = result.get('_collection', 'unknown')
            
            print(f"[{idx}] Score: {distance:.4f} | Collection: {coll}")
            print(f"    Doc ID:  {entity.get('doc_id', 'N/A')}")
            print(f"    Title:   {entity.get('title', 'N/A')[:80]}")
            print(f"    Date:    {entity.get('date', 'N/A')} | Topic: {entity.get('topic', 'N/A')}")
            print(f"    Source:  {entity.get('source', 'N/A')}")
            
            if entity.get('has_image'):
                img_path = entity.get('image_path', '')
                if img_path:
                    # 이미지 경로 표시 (클릭 가능한 링크)
                    abs_path = Path(img_path).resolve()
                    display_name = abs_path.name
                    print(f"    Image:   🖼️  {display_name}")
                    if abs_path.exists():
                        # 클릭 가능한 링크 (OSC 8 hyperlink)
                        print(f"             📁 \033]8;;file://{abs_path}\033\\{abs_path}\033]8;;\033\\")
                    else:
                        print(f"             ⚠️  파일 없음: {abs_path}")
                else:
                    print(f"    Image:   🖼️  N/A")
            else:
                print(f"    Image:   ❌ No image")
            
            content = entity.get('content', '')
            if content:
                preview = content[:150].replace('\n', ' ')
                print(f"    Preview: {preview}...")
            print()
        
        # 인터랙티브 모드: 전체 내용 보기 옵션
        if interactive and results:
            print(f"{'─'*80}")
            print("💡 옵션:")
            print("   • 전체 내용 보기: 결과 번호 입력 (예: 1, 2, 3 또는 1,3,5)")
            print("   • 이미지 링크 보기: i+번호 입력 (예: i1, i2, i3 또는 i1,i3)")
            print("     → 클릭 가능한 파일 링크가 표시됩니다")
            print("   • 종료: Enter 키")
            print(f"{'─'*80}\n")
            
            try:
                user_input = input("입력: ").strip()
                
                if user_input:
                    # 쉼표로 구분된 입력들 파싱
                    items = [x.strip() for x in user_input.split(',')]
                    
                    for item in items:
                        # 이미지 링크 표시 (i1, i2 등)
                        if item.lower().startswith('i') and len(item) > 1:
                            try:
                                idx = int(item[1:])
                                if 1 <= idx <= len(results):
                                    entity = results[idx-1].get('entity', {})
                                    if entity.get('has_image'):
                                        img_path = entity.get('image_path', '')
                                        if img_path:
                                            abs_path = Path(img_path).resolve()
                                            if abs_path.exists():
                                                print(f"\n[{idx}] 이미지 정보:")
                                                self._show_clickable_image_link(str(abs_path))
                                            else:
                                                print(f"\n⚠️  [{idx}] 이미지 파일을 찾을 수 없습니다")
                                        else:
                                            print(f"\n⚠️  [{idx}] 이미지 경로 정보가 없습니다")
                                    else:
                                        print(f"\n⚠️  [{idx}] 이미지가 없는 결과입니다")
                                else:
                                    print(f"\n⚠️  잘못된 번호: {item} (유효 범위: 1-{len(results)})")
                            except ValueError:
                                print(f"\n⚠️  올바른 형식이 아닙니다: {item} (예: i1, i2)")
                        
                        # 전체 내용 보기 (숫자만)
                        else:
                            try:
                                idx = int(item)
                                if 1 <= idx <= len(results):
                                    self._show_full_content(results[idx-1], idx)
                                else:
                                    print(f"\n⚠️  잘못된 번호: {idx} (유효 범위: 1-{len(results)})")
                            except ValueError:
                                print(f"\n⚠️  올바른 입력이 아닙니다: {item} (예: 1 또는 i1)")
            except KeyboardInterrupt:
                print("\n")
    
    def _show_clickable_image_link(self, image_path: str) -> bool:
        """클릭 가능한 이미지 링크 출력"""
        try:
            abs_path = Path(image_path).resolve()
            if not abs_path.exists():
                print(f"⚠️  이미지 파일을 찾을 수 없습니다: {abs_path}")
                return False
            
            # 파일 크기 정보
            file_size = abs_path.stat().st_size / 1024  # KB
            
            print(f"\n{'='*80}")
            print(f"🖼️  이미지 정보")
            print(f"{'='*80}")
            print(f"파일명: {abs_path.name}")
            print(f"크기: {file_size:.1f} KB")
            print(f"\n{'─'*80}")
            print("📎 다음 링크를 클릭하여 이미지를 여세요:")
            print(f"   \033]8;;file://{abs_path}\033\\file://{abs_path}\033]8;;\033\\")
            print(f"\n또는 경로 복사:")
            print(f"   {abs_path}")
            print(f"\n터미널에서 직접 열기:")
            print(f"   xdg-open '{abs_path}'")
            print(f"{'─'*80}\n")
            
            logger.info(f"Image link displayed: {abs_path.name}")
            
            return True
        
        except Exception as e:
            print(f"⚠️  이미지 정보를 가져오는 중 오류 발생: {e}")
            return False
    
    def _show_full_content(self, result: Dict[str, Any], idx: int):
        """단일 결과의 전체 내용 표시"""
        entity = result.get('entity', {})
        
        print(f"\n{'='*80}")
        print(f"📰 결과 [{idx}] 전체 내용")
        print(f"{'='*80}\n")
        
        print(f"Doc ID:  {entity.get('doc_id', 'N/A')}")
        print(f"Title:   {entity.get('title', 'N/A')}")
        print(f"Date:    {entity.get('date', 'N/A')}")
        print(f"Source:  {entity.get('source', 'N/A')}")
        
        img_path = None
        if entity.get('has_image'):
            img_path = entity.get('image_path', '')
            if img_path:
                abs_path = Path(img_path).resolve()
                print(f"Image:   🖼️  {abs_path}")
                if abs_path.exists():
                    print(f"         파일 크기: {abs_path.stat().st_size / 1024:.1f} KB")
        
        print(f"\n{'-'*80}")
        print("Content:")
        print(f"{'-'*80}\n")
        
        content = entity.get('content', 'No content available')
        print(content)
        
        print(f"\n{'='*80}")
        
        # 이미지가 있으면 클릭 가능한 링크 보여주기
        if img_path:
            abs_path = Path(img_path).resolve()
            if abs_path.exists():
                try:
                    response = input(f"\n🖼️  이미지 링크를 표시하시겠습니까? [y/N]: ").strip().lower()
                    if response in ['y', 'yes', 'ㅛ']:  # ㅛ는 한글 키보드 y
                        self._show_clickable_image_link(str(abs_path))
                except KeyboardInterrupt:
                    print()
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Multimodal RAG Search Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single collection search
  python run_search.py --mode text --query "화재 사건" -c fire_multimodal --db-file db/fire.db

  # Multiple collections search (comma-separated)
  python run_search.py --mode text --query "화재 대피" -c fire_news,disaster_manual --db-file db/fire.db

  # Image search
  python run_search.py --mode image --image data/query_images/fire/fire1.jpg -c fire_multimodal --db-file db/fire.db

  # Hybrid search
  python run_search.py --mode hybrid --query "대형 화재" --image data/query_images/fire/fire2.jpg -c fire_multimodal --db-file db/fire.db

  # With filters
  python run_search.py --mode text --query "화재" -c disaster_manual --topic fire --db-file db/fire.db
        """
    )
    
    parser.add_argument('--mode', required=True, choices=['text', 'image', 'hybrid'],
                       help='Search mode: text, image, or hybrid')
    parser.add_argument('--collection', '-c', required=True,
                       help='Collection name(s), comma-separated for multiple (e.g., fire_news,disaster_manual)')
    parser.add_argument('--query', '-q',
                       help='Search query (required for text/hybrid mode)')
    parser.add_argument('--image', '-i',
                       help='Query image path (required for image/hybrid mode)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results to return')
    parser.add_argument('--db-file', default='./fire_multimodal.db',
                       help='Milvus database file')
    parser.add_argument('--date-start',
                       help='Start date filter (YYYYMMDD format, e.g., 20210101)')
    parser.add_argument('--date-end',
                       help='End date filter (YYYYMMDD format, e.g., 20231231)')
    parser.add_argument('--category',
                       help='Category filter (e.g., disaster)')
    parser.add_argument('--topic',
                       help='Topic filter (e.g., fire)')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Disable interactive mode (no full content viewing)')
    parser.add_argument('--generate', '-g', action='store_true',
                       help='Generate answer using LLM (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)')
    parser.add_argument('--llm', default='openai', choices=['openai', 'anthropic'],
                       help='LLM provider for answer generation (default: openai)')
    parser.add_argument('--model',
                       help='LLM model name (default: gpt-4o-mini for OpenAI, claude-sonnet-4-20250514 for Anthropic)')
    
    args = parser.parse_args()
    
    # 검증
    if args.mode in ['text', 'hybrid'] and not args.query:
        parser.error(f"--query is required for {args.mode} mode")
    
    if args.mode in ['image', 'hybrid'] and not args.image:
        parser.error(f"--image is required for {args.mode} mode")
    
    # 다중 컬렉션 파싱
    collections = [c.strip() for c in args.collection.split(',')]
    
    print("\n" + "="*80)
    print("🔍 Multimodal RAG Search")
    print("="*80)
    print(f"Mode:        {args.mode}")
    print(f"Collections: {collections}")
    print(f"Database:    {args.db_file}")
    if args.query:
        print(f"Query:       {args.query}")
    if args.image:
        print(f"Image:       {args.image}")
    print(f"Top-K:       {args.top_k}")
    print("="*80)
    
    try:
        # Searcher 초기화 (다중 컬렉션 지원)
        searcher = MultimodalSearcher(
            db_file=args.db_file,
            collections=collections
        )
        
        # Interactive 모드 설정
        interactive = not args.non_interactive
        
        # 필터 정보 출력
        if args.date_start or args.date_end or args.category or args.topic:
            print("\n📅 Applying Filters:")
            if args.date_start:
                print(f"   Start Date: {args.date_start}")
            if args.date_end:
                print(f"   End Date: {args.date_end}")
            if args.category:
                print(f"   Category: {args.category}")
            if args.topic:
                print(f"   Topic: {args.topic}")
            print()
        
        # 검색 수행
        if args.mode == 'text':
            results = searcher.search_by_text(
                args.query, 
                top_k=args.top_k,
                date_start=args.date_start,
                date_end=args.date_end,
                category=args.category,
                topic=args.topic
            )
            searcher.print_results(results, 'text', interactive=interactive)
        
        elif args.mode == 'image':
            results = searcher.search_by_image(
                args.image, 
                top_k=args.top_k,
                date_start=args.date_start,
                date_end=args.date_end,
                category=args.category,
                topic=args.topic
            )
            searcher.print_results(results, 'image', interactive=interactive)
        
        elif args.mode == 'hybrid':
            # 텍스트와 이미지 검색 결과를 결합
            text_results = searcher.search_by_text(
                args.query, 
                top_k=args.top_k * 2,
                date_start=args.date_start,
                date_end=args.date_end,
                category=args.category,
                topic=args.topic
            )
            image_results = searcher.search_by_image(
                args.image, 
                top_k=args.top_k * 2,
                date_start=args.date_start,
                date_end=args.date_end,
                category=args.category,
                topic=args.topic
            )
            
            # 간단한 결합 (doc_id 기반)
            combined = {}
            for r in text_results:
                doc_id = r.get('entity', {}).get('doc_id')
                if doc_id:
                    combined[doc_id] = r
            
            for r in image_results:
                doc_id = r.get('entity', {}).get('doc_id')
                if doc_id and doc_id in combined:
                    # 이미 있으면 스코어 향상
                    combined[doc_id]['distance'] = (combined[doc_id]['distance'] + r['distance']) / 2
                elif doc_id:
                    combined[doc_id] = r
            
            # 스코어로 정렬
            results = sorted(combined.values(), key=lambda x: x['distance'], reverse=True)[:args.top_k]
            searcher.print_results(results, 'hybrid', interactive=interactive)
        
        # 답변 생성 (선택사항)
        if args.generate and results:
            print(f"\n{'='*80}")
            print("🤖 Generating Answer with LLM...")
            print(f"{'='*80}\n")
            
            generator = AnswerGenerator(llm=args.llm, model=args.model)
            query_text = args.query if args.query else "이미지 검색 결과 요약"
            answer = generator.generate(query_text, results)
            
            print(answer)
            print(f"\n{'='*80}")
        
        print(f"\n✅ Search completed successfully!")
        print()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

