#!/usr/bin/env python3
"""
이미지 파일명 기반 자동 매칭 및 멀티모달 DB 생성

이미지 파일명이 doc_id인 경우, 해당 뉴스와 자동으로 매칭하여
멀티모달 Milvus DB를 생성합니다.

사용법:
    # 전체 뉴스 사용
    python create_multimodal_db_from_images.py \
      --news news_data/01_disaster_Fire_3years.json \
      --images naver_news_images/fire \
      --collection fire_multimodal \
      --news-range fire_all
    
    # 클러스터된 뉴스만 사용
    python create_multimodal_db_from_images.py \
      --news news_data/01_disaster_Fire_3years.json \
      --images naver_news_images/fire \
      --collection fire_multimodal \
      --news-range fire_clustered \
      --clustered-csv clustered_news.csv
"""

import json
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
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
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("transformers not installed. Install with: pip install transformers torch pillow")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalDBCreator:
    """이미지 파일명 기반 자동 매칭 멀티모달 DB 생성기"""
    
    def __init__(
        self,
        db_file: str = "./multimodal.db",
        text_embedding_model: str = "jhgan/ko-sroberta-multitask",
        clip_model: str = "openai/clip-vit-base-patch32"
    ):
        """
        초기화
        
        Args:
            db_file: Milvus Lite 데이터베이스 파일
            text_embedding_model: 텍스트 임베딩 모델
            clip_model: CLIP 모델
        """
        self.db_file = db_file
        
        # Milvus Client 초기화
        self.client = MilvusClient(db_file)
        logger.info(f"✅ Connected to Milvus Lite at {db_file}")
        
        # Text embedding 모델 초기화
        logger.info(f"Loading text embedding model: {text_embedding_model}")
        device = 'cuda' if self._check_cuda() else 'cpu'
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name=text_embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"✅ Text embedding model loaded on {device}")
        
        # CLIP 모델 초기화
        if CLIP_AVAILABLE:
            logger.info(f"Loading CLIP model: {clip_model}")
            self.device = "cuda" if self._check_cuda() else "cpu"
            self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
            logger.info(f"✅ CLIP model loaded on {self.device}")
        else:
            raise ImportError("CLIP is required. Install with: pip install transformers torch pillow")
    
    def _check_cuda(self) -> bool:
        """CUDA 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load_clustered_doc_ids(self, csv_file: str) -> Set[str]:
        """
        클러스터된 뉴스의 doc_id 로드
        
        Args:
            csv_file: clustered_news.csv 파일 경로
        
        Returns:
            클러스터된 doc_id 집합
        """
        logger.info(f"Loading clustered doc_ids from {csv_file}")
        doc_ids = set()
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get('doc_id', '').strip()
                if doc_id:
                    doc_ids.add(doc_id)
        
        logger.info(f"✅ Loaded {len(doc_ids)} clustered doc_ids")
        return doc_ids
    
    def load_news(self, news_file: str, news_range: str = 'fire_all', 
                  clustered_csv: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        뉴스 데이터 로드
        
        Args:
            news_file: 뉴스 JSON 파일
            news_range: 'fire_all' 또는 'fire_clustered'
            clustered_csv: 클러스터 CSV 파일 (fire_clustered인 경우 필수)
        
        Returns:
            뉴스 리스트
        """
        logger.info(f"Loading news from {news_file} (range: {news_range})")
        
        with open(news_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 데이터 구조 파싱
        news_list = []
        if isinstance(raw_data, list) and len(raw_data) > 0:
            first_item = raw_data[0]
            
            if "item" in first_item and "documentList" in first_item["item"]:
                # item.documentList 구조
                for item in raw_data:
                    if "item" in item and "documentList" in item["item"]:
                        for doc in item["item"]["documentList"]:
                            news_list.append({
                                "doc_id": doc.get("docID", ""),
                                "title": doc.get("title", ""),
                                "text": doc.get("content", ""),
                                "date": doc.get("date", ""),
                                "url": doc.get("url", ""),
                                "source": doc.get("writerName", ""),
                                "category": "disaster",
                                "topic": "fire"
                            })
            elif "search_result" in first_item:
                # search_result 구조
                for item in raw_data:
                    if "search_result" in item:
                        news_list.extend(item["search_result"])
            else:
                # 이미 표준화된 구조
                news_list = raw_data
        
        logger.info(f"Loaded {len(news_list)} total news articles")
        
        # 클러스터된 뉴스만 필터링
        if news_range == 'fire_clustered':
            if not clustered_csv:
                raise ValueError("clustered_csv is required when news_range='fire_clustered'")
            
            clustered_doc_ids = self.load_clustered_doc_ids(clustered_csv)
            filtered_news = [n for n in news_list if n.get("doc_id") in clustered_doc_ids]
            logger.info(f"Filtered to {len(filtered_news)} clustered news articles")
            return filtered_news
        
        return news_list
    
    def load_images_with_doc_ids(self, image_dir: str, 
                                 valid_doc_ids: Optional[Set[str]] = None) -> Dict[str, str]:
        """
        이미지 파일을 로드하고 파일명에서 doc_id 추출
        
        Args:
            image_dir: 이미지 디렉토리
            valid_doc_ids: 유효한 doc_id 집합 (필터링용)
        
        Returns:
            {doc_id: image_path} 매핑
        """
        logger.info(f"Loading images from {image_dir}")
        image_path = Path(image_dir)
        
        if not image_path.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        image_mappings = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
        
        for file_path in image_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # 파일명에서 doc_id 추출 (확장자 제거)
                doc_id = file_path.stem
                
                # 유효한 doc_id인지 확인
                if valid_doc_ids is None or doc_id in valid_doc_ids:
                    image_mappings[doc_id] = str(file_path)
        
        logger.info(f"✅ Found {len(image_mappings)} images matching doc_ids")
        return image_mappings
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """이미지 로드"""
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def encode_image(self, image: Image.Image) -> List[float]:
        """CLIP을 사용하여 이미지 임베딩 생성"""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None
    
    def create_collection(self, collection_name: str):
        """멀티모달 컬렉션 생성"""
        # 기존 컬렉션 삭제
        if self.client.has_collection(collection_name):
            logger.info(f"Dropping existing collection '{collection_name}'")
            self.client.drop_collection(collection_name)
        
        # 새 컬렉션 생성
        # 텍스트 임베딩 (768차원)과 이미지 임베딩 (512차원)을 별도로 저장
        self.client.create_collection(
            collection_name=collection_name,
            dimension=768,  # 텍스트 임베딩 차원
            metric_type="COSINE"
        )
        
        logger.info(f"✅ Created collection '{collection_name}'")
    
    def prepare_multimodal_data(
        self,
        news_list: List[Dict[str, Any]],
        image_mappings: Dict[str, str],
        batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        멀티모달 데이터 준비
        
        Args:
            news_list: 뉴스 리스트
            image_mappings: {doc_id: image_path} 매핑
            batch_size: 배치 크기
        
        Returns:
            Milvus 삽입용 데이터 리스트
        """
        logger.info(f"Preparing multimodal data for {len(news_list)} news articles...")
        
        # 텍스트 임베딩 생성 (배치)
        texts = [n.get("text", "") for n in news_list]
        logger.info("Generating text embeddings...")
        
        all_text_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Text embedding"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.text_embeddings.embed_documents(batch_texts)
            all_text_embeddings.extend(batch_embeddings)
        
        # 데이터 구성
        milvus_data = []
        images_with_embeddings = 0
        
        for idx, (news, text_embedding) in enumerate(zip(news_list, all_text_embeddings)):
            doc_id = news.get("doc_id", "")
            
            # 이미지 임베딩 생성
            image_embedding = None
            has_image = False
            image_path = ""
            
            if doc_id in image_mappings:
                img_path = image_mappings[doc_id]
                image = self.load_image(img_path)
                
                if image is not None:
                    image_embedding = self.encode_image(image)
                    if image_embedding is not None:
                        has_image = True
                        image_path = img_path
                        images_with_embeddings += 1
            
            # 이미지 임베딩이 없으면 제로 벡터 (512차원)
            if image_embedding is None:
                image_embedding = [0.0] * 512
            
            # Milvus 데이터 항목
            data_item = {
                "id": idx,
                "vector": text_embedding,  # 텍스트 임베딩 (768차원)
                "doc_id": doc_id[:200],
                "title": news.get("title", "")[:500],
                "content": news.get("text", "")[:5000],
                "date": news.get("date", "")[:20],
                "url": news.get("url", "")[:500],
                "source": news.get("source", "")[:200],
                "category": news.get("category", "disaster")[:50],
                "topic": news.get("topic", "fire")[:100],
                "has_image": has_image,
                "image_path": image_path[:500],
                "image_embedding": json.dumps(image_embedding)  # JSON 문자열로 저장
            }
            
            milvus_data.append(data_item)
        
        logger.info(f"✅ Prepared {len(milvus_data)} items ({images_with_embeddings} with images)")
        return milvus_data
    
    def insert_data(self, collection_name: str, data: List[Dict[str, Any]], 
                   batch_size: int = 1000):
        """데이터 삽입"""
        if not data:
            logger.warning("No data to insert")
            return False
        
        try:
            logger.info(f"Inserting {len(data)} items to collection '{collection_name}'...")
            
            for i in tqdm(range(0, len(data), batch_size), desc="Inserting"):
                batch_data = data[i:i+batch_size]
                self.client.insert(
                    collection_name=collection_name,
                    data=batch_data
                )
            
            logger.info(f"✅ Successfully inserted {len(data)} items")
            return True
        
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_statistics(self, news_list: List[Dict[str, Any]], 
                        image_mappings: Dict[str, str],
                        milvus_data: List[Dict[str, Any]]):
        """통계 출력"""
        images_with_embeddings = sum(1 for item in milvus_data if item["has_image"])
        
        print("\n" + "="*70)
        print("📊 Multimodal DB Creation Statistics")
        print("="*70)
        print(f"Total news articles:     {len(news_list)}")
        print(f"Available images:        {len(image_mappings)}")
        print(f"News with images:        {images_with_embeddings}")
        print(f"News without images:     {len(news_list) - images_with_embeddings}")
        print(f"Total items in DB:       {len(milvus_data)}")
        print(f"Database file:           {self.db_file}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Create multimodal DB from image filenames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 뉴스 사용
  python create_multimodal_db_from_images.py \\
    --news news_data/01_disaster_Fire_3years.json \\
    --images naver_news_images/fire \\
    --collection fire_multimodal \\
    --news-range fire_all

  # 클러스터된 뉴스만 사용
  python create_multimodal_db_from_images.py \\
    --news news_data/01_disaster_Fire_3years.json \\
    --images naver_news_images/fire \\
    --collection fire_multimodal \\
    --news-range fire_clustered \\
    --clustered-csv clustered_news.csv
        """
    )
    
    parser.add_argument('--news', '-n', required=True,
                       help='News JSON file')
    parser.add_argument('--images', '-i', required=True,
                       help='Image directory (filenames should be doc_ids)')
    parser.add_argument('--collection', '-c', required=True,
                       help='Milvus collection name')
    parser.add_argument('--news-range', choices=['fire_all', 'fire_clustered'],
                       default='fire_all',
                       help='News range: fire_all (all news) or fire_clustered (clustered only)')
    parser.add_argument('--clustered-csv',
                       help='Clustered news CSV file (required when news-range=fire_clustered)')
    parser.add_argument('--db-file', default='./multimodal.db',
                       help='Milvus Lite database file')
    parser.add_argument('--text-model', default='jhgan/ko-sroberta-multitask',
                       help='Text embedding model')
    parser.add_argument('--clip-model', default='openai/clip-vit-base-patch32',
                       help='CLIP model for image embeddings')
    
    args = parser.parse_args()
    
    # 검증
    if args.news_range == 'fire_clustered' and not args.clustered_csv:
        parser.error("--clustered-csv is required when --news-range=fire_clustered")
    
    print("\n" + "="*70)
    print("🎯 Multimodal DB Creator (Image Filename Based)")
    print("="*70)
    print(f"News file:       {args.news}")
    print(f"Image directory: {args.images}")
    print(f"Collection:      {args.collection}")
    print(f"News range:      {args.news_range}")
    if args.clustered_csv:
        print(f"Clustered CSV:   {args.clustered_csv}")
    print(f"Database file:   {args.db_file}")
    print("="*70 + "\n")
    
    try:
        # Creator 초기화
        creator = MultimodalDBCreator(
            db_file=args.db_file,
            text_embedding_model=args.text_model,
            clip_model=args.clip_model
        )
        
        # 1. 뉴스 로드
        news_list = creator.load_news(
            args.news,
            news_range=args.news_range,
            clustered_csv=args.clustered_csv
        )
        
        if not news_list:
            logger.error("No news data loaded!")
            return 1
        
        # 2. 유효한 doc_id 집합 생성
        valid_doc_ids = {n.get("doc_id") for n in news_list if n.get("doc_id")}
        
        # 3. 이미지 로드 (파일명 = doc_id)
        image_mappings = creator.load_images_with_doc_ids(
            args.images,
            valid_doc_ids=valid_doc_ids
        )
        
        # 4. 컬렉션 생성
        creator.create_collection(args.collection)
        
        # 5. 멀티모달 데이터 준비
        milvus_data = creator.prepare_multimodal_data(news_list, image_mappings)
        
        # 6. 데이터 삽입
        success = creator.insert_data(args.collection, milvus_data)
        
        if success:
            # 7. 통계 출력
            creator.print_statistics(news_list, image_mappings, milvus_data)
            
            print("\n✅ Multimodal DB creation completed successfully!")
            print(f"\n💡 Next steps:")
            print(f"   # Text search")
            print(f"   python multimodal_rag_v2.py --mode search --collection {args.collection} --query '화재 사건'")
            print(f"\n   # Image search")
            print(f"   python multimodal_rag_v2.py --mode search-image --collection {args.collection} --image IMAGE_PATH")
            print(f"\n   # Hybrid search")
            print(f"   python multimodal_rag_v2.py --mode hybrid --collection {args.collection} --query '화재' --image IMAGE_PATH")
            print()
            
            return 0
        else:
            logger.error("Failed to create multimodal DB")
            return 1
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

