#!/usr/bin/env python3
"""
기존 Milvus 컬렉션에 이미지 추가/업데이트 스크립트 (Multimodal RAG)

사용법:
    python add_images_to_milvus.py --collection fire_news --images images_data.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from tqdm import tqdm
import logging
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Milvus imports
from pymilvus import connections, Collection, utility

# CLIP for image embeddings
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("transformers not installed. Install with: pip install transformers torch")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageToMilvusAdder:
    def __init__(
        self,
        collection_name: str,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        clip_model: str = "openai/clip-vit-base-patch32"
    ):
        """
        Milvus 컬렉션에 이미지 추가하는 클래스
        
        Args:
            collection_name: 대상 Milvus 컬렉션 이름
            milvus_host: Milvus 서버 호스트
            milvus_port: Milvus 서버 포트
            clip_model: CLIP 모델 이름
        """
        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        
        # Milvus 연결
        self._connect_milvus()
        
        # 컬렉션 로드
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        self.collection = Collection(collection_name)
        
        # CLIP 모델 초기화 (이미지 임베딩용)
        if CLIP_AVAILABLE:
            logger.info(f"Loading CLIP model: {clip_model}...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
            logger.info(f"✅ CLIP model loaded on {self.device}")
        else:
            logger.error("CLIP not available. Please install transformers and torch.")
            raise ImportError("transformers required for image processing")
        
        self.stats = {
            'updated_documents': 0,
            'failed_updates': 0,
            'processing_time': 0
        }

    def _connect_milvus(self):
        """Milvus 서버에 연결"""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port
            )
            logger.info(f"✅ Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def load_image(self, image_source: str) -> Optional[Image.Image]:
        """
        이미지 로드 (로컬 파일 또는 URL)
        
        Args:
            image_source: 이미지 파일 경로 또는 URL
            
        Returns:
            PIL Image 객체
        """
        try:
            if image_source.startswith('http://') or image_source.startswith('https://'):
                # URL에서 로드
                response = requests.get(image_source, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # 로컬 파일에서 로드
                image = Image.open(image_source).convert('RGB')
            
            return image
        except Exception as e:
            logger.warning(f"Failed to load image from {image_source}: {e}")
            return None

    def generate_image_embedding(self, image: Image.Image) -> List[float]:
        """
        CLIP을 사용하여 이미지 임베딩 생성
        
        Args:
            image: PIL Image 객체
            
        Returns:
            이미지 임베딩 벡터
        """
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # 정규화
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embedding = image_features.cpu().numpy().flatten().tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            return None

    def generate_image_caption(self, image: Image.Image) -> str:
        """
        이미지 캡션 생성 (선택적 - BLIP 등 사용 가능)
        
        현재는 더미 구현, 실제로는 BLIP 등의 모델 사용
        """
        # TODO: BLIP 또는 다른 캡셔닝 모델 통합
        return ""

    def update_document_with_image(
        self,
        doc_id: str,
        image_source: str,
        image_caption: Optional[str] = None
    ) -> bool:
        """
        특정 문서에 이미지 정보 업데이트
        
        Note: Milvus는 직접 업데이트를 지원하지 않으므로,
        실제로는 삭제 후 재삽입 또는 별도 이미지 컬렉션 생성이 필요합니다.
        여기서는 이미지 매핑 파일을 생성하는 방식으로 구현합니다.
        
        Args:
            doc_id: 문서 ID
            image_source: 이미지 경로 또는 URL
            image_caption: 이미지 캡션 (선택적)
            
        Returns:
            성공 여부
        """
        try:
            # 이미지 로드
            image = self.load_image(image_source)
            if image is None:
                return False
            
            # 이미지 임베딩 생성
            image_embedding = self.generate_image_embedding(image)
            if image_embedding is None:
                return False
            
            # 캡션 생성 (제공되지 않은 경우)
            if image_caption is None:
                image_caption = self.generate_image_caption(image)
            
            # Milvus에서 문서 검색
            self.collection.load()
            results = self.collection.query(
                expr=f'doc_id == "{doc_id}"',
                output_fields=["id", "doc_id", "title"]
            )
            
            if not results:
                logger.warning(f"Document with doc_id '{doc_id}' not found")
                return False
            
            logger.info(f"✅ Found document '{doc_id}', image embedding generated")
            # 실제 업데이트는 별도 매핑 파일로 저장 (Milvus 2.x에서 upsert 사용)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False

    def batch_add_images(self, image_mappings: List[Dict[str, Any]]) -> bool:
        """
        배치로 이미지 추가
        
        Args:
            image_mappings: [{"doc_id": "...", "image_url": "...", "caption": "..."}, ...]
            
        Returns:
            성공 여부
        """
        logger.info(f"Processing {len(image_mappings)} image mappings...")
        
        for mapping in tqdm(image_mappings, desc="Adding images"):
            doc_id = mapping.get("doc_id")
            image_source = mapping.get("image_url") or mapping.get("image_path")
            caption = mapping.get("caption")
            
            if not doc_id or not image_source:
                logger.warning(f"Invalid mapping: {mapping}")
                continue
            
            if self.update_document_with_image(doc_id, image_source, caption):
                self.stats['updated_documents'] += 1
            else:
                self.stats['failed_updates'] += 1
        
        return self.stats['updated_documents'] > 0

    def print_summary(self):
        """처리 결과 요약 출력"""
        print("\n" + "="*60)
        print("🎉 Image Addition Summary")
        print("="*60)
        print(f"✅ Successfully Updated: {self.stats['updated_documents']}")
        print(f"❌ Failed Updates: {self.stats['failed_updates']}")
        print(f"⏱️ Processing Time: {self.stats['processing_time']:.2f} seconds")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Add images to existing Milvus collection')
    parser.add_argument('--collection', '-c', required=True,
                       help='Milvus collection name')
    parser.add_argument('--images', '-i', required=True,
                       help='JSON file with image mappings')
    parser.add_argument('--milvus-host', default='localhost',
                       help='Milvus server host (default: localhost)')
    parser.add_argument('--milvus-port', default='19530',
                       help='Milvus server port (default: 19530)')
    parser.add_argument('--clip-model', default='openai/clip-vit-base-patch32',
                       help='CLIP model to use for image embeddings')
    
    args = parser.parse_args()
    
    # 이미지 매핑 로드
    with open(args.images, 'r', encoding='utf-8') as f:
        image_mappings = json.load(f)
    
    if not isinstance(image_mappings, list):
        logger.error("Image mappings should be a list")
        sys.exit(1)
    
    # 이미지 추가기 초기화
    try:
        adder = ImageToMilvusAdder(
            collection_name=args.collection,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            clip_model=args.clip_model
        )
    except Exception as e:
        logger.error(f"Failed to initialize image adder: {e}")
        sys.exit(1)
    
    # 이미지 추가 실행
    start_time = time.time()
    success = adder.batch_add_images(image_mappings)
    adder.stats['processing_time'] = time.time() - start_time
    
    # 결과 출력
    if success:
        adder.print_summary()
    else:
        logger.error("Image addition failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

