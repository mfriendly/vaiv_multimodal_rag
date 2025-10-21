#!/usr/bin/env python3
"""
뉴스 데이터 → Milvus Lite 변환 스크립트 V2 (공식 API 사용)

사용법:
    python convert_news_to_milvus_lite_v2.py --input news_data/01_disaster_Fire_3years.json --collection fire_news
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
from tqdm import tqdm
import logging

# Milvus Client (High-level API)
from pymilvus import MilvusClient

# LangChain imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI for metadata extraction
import openai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsToMilvusConverterV2:
    def __init__(
        self,
        db_file: str = "./milvus_lite.db",
        openai_api_key: Optional[str] = None,
        text_embedding_dim: int = 768,
    ):
        """
        뉴스 데이터를 Milvus Lite로 변환하는 클래스 (공식 API 사용)
        
        Args:
            db_file: Milvus Lite 데이터베이스 파일 경로
            openai_api_key: GPT를 사용한 메타데이터 추출용 API 키
            text_embedding_dim: 텍스트 임베딩 차원
        """
        self.db_file = db_file
        self.text_embedding_dim = text_embedding_dim
        
        # Milvus Client 초기화 (High-level API!)
        self.client = MilvusClient(db_file)
        logger.info(f"✅ Connected to Milvus Lite at {db_file}")
        
        # HuggingFace 임베딩 모델 초기화
        logger.info("Loading HuggingFace embeddings model for text...")
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda' if self._check_cuda() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # OpenAI 클라이언트 설정
        self.openai_client = None
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized for metadata extraction")
        
        # 통계 정보
        self.stats = {
            'total_documents': 0,
            'processed_documents': 0,
            'created_collections': [],
            'processing_time': 0
        }

    def _check_cuda(self) -> bool:
        """CUDA 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def create_collection(self, collection_name: str):
        """
        컬렉션 생성 (MilvusClient API 사용)
        """
        # 기존 컬렉션이 있으면 삭제
        if self.client.has_collection(collection_name):
            logger.info(f"Dropping existing collection '{collection_name}'...")
            self.client.drop_collection(collection_name)
        
        # 새 컬렉션 생성 (간단한 API!)
        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.text_embedding_dim,
            metric_type="COSINE",
            # auto_id=True,  # Primary key 자동 생성
        )
        
        logger.info(f"✅ Created collection '{collection_name}' with dimension={self.text_embedding_dim}")
        self.stats['created_collections'].append(collection_name)

    def load_news_data(self, file_path: str) -> List[Dict[str, Any]]:
        """뉴스 JSON 파일을 로드하고 표준화된 형태로 변환"""
        logger.info(f"Loading news data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 데이터 구조 분석 및 변환
        if isinstance(raw_data, list) and len(raw_data) > 0:
            first_item = raw_data[0]
            
            # Case 1: search_result 구조
            if "search_result" in first_item:
                logger.info("Detected search_result structure")
                news_data = []
                for item in raw_data:
                    if "search_result" in item:
                        news_data.extend(item["search_result"])
                return news_data
            
            # Case 2: item.documentList 구조
            elif "item" in first_item and "documentList" in first_item["item"]:
                logger.info("Detected item.documentList structure")
                news_data = []
                for item in raw_data:
                    if "item" in item and "documentList" in item["item"]:
                        document_list = item["item"]["documentList"]
                        for doc in document_list:
                            standardized_doc = {
                                "date": doc.get("date", ""),
                                "title": doc.get("title", ""),
                                "text": doc.get("content", ""),
                                "doc_id": doc.get("docID", ""),
                                "url": doc.get("url", ""),
                                "source": doc.get("writerName", ""),
                                "vks": doc.get("vks", [])
                            }
                            news_data.append(standardized_doc)
                return news_data
            
            # Case 3: 이미 표준화된 구조
            else:
                logger.info("Detected standardized structure")
                return raw_data
        
        logger.warning(f"Unknown data structure in {file_path}")
        return []

    def extract_metadata_from_filename(self, filename: str) -> Tuple[str, str]:
        """파일명에서 카테고리와 토픽 추출"""
        filename_lower = filename.lower()
        
        if "disaster" in filename_lower:
            category = "disaster"
        elif "crime" in filename_lower:
            category = "crime"
        else:
            category = "other"
        
        topic = "unknown"
        topic_mapping = {
            "fire": "fire", "crime": "crime", "snow": "heavy snow",
            "earthquake": "earthquake", "infection": "infection",
            "traffic": "traffic accident", "rain": "heavy rain",
            "heatwave": "heatwave", "landslide": "landslide",
            "storm": "storm", "pm10": "pm10",
            "water": "water accident", "density": "density"
        }
        
        for key, value in topic_mapping.items():
            if key in filename_lower:
                topic = value
                break
        
        return category, topic

    def prepare_milvus_data(
        self,
        news_data: List[Dict],
        filename: str,
        use_gpt_metadata: bool = True,
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        뉴스 데이터를 Milvus 삽입 형식으로 변환
        
        Returns:
            List of dicts: [{"id": ..., "vector": [...], "title": ..., ...}, ...]
        """
        file_category, file_topic = self.extract_metadata_from_filename(filename)
        
        logger.info(f"Preparing Milvus data from {len(news_data)} news items...")
        
        texts_to_embed = []
        valid_news = []
        
        # 유효한 뉴스 필터링
        for idx, news in enumerate(news_data):
            title = news.get("title", "")
            content = news.get("text", "")
            
            if len(content.strip()) < 50:
                continue
            
            valid_news.append((idx, news, title, content))
            texts_to_embed.append(content)
        
        if not valid_news:
            logger.warning("No valid news items to process")
            return []
        
        # 배치 임베딩 생성
        logger.info(f"Generating embeddings for {len(texts_to_embed)} documents...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding batches"):
            if 0: #i>20000:
                break
            batch_texts = texts_to_embed[i:i+batch_size]
            batch_embeddings = self.text_embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Milvus 데이터 구성 (딕셔너리 리스트 형식!)
        logger.info("Preparing data for Milvus insertion...")
        current_timestamp = int(time.time())
        
        milvus_data = []
        for embedding, (idx, news, title, content) in zip(all_embeddings, valid_news):
            if use_gpt_metadata and self.openai_client:
                # GPT 메타데이터는 나중에 구현
                category, topic = file_category, file_topic
            else:
                category, topic = file_category, file_topic
            
            date = news.get("date", "")
            doc_id = news.get("doc_id", f"{filename}_{idx}")
            url = news.get("url", "")
            source = news.get("source", "")
            
            # Milvus 데이터 항목 (딕셔너리!)
            data_item = {
                "id": idx,  # Primary key (integer)
                "vector": embedding,  # 벡터는 그대로 list
                "doc_id": doc_id[:200] if doc_id else "",
                "title": title[:500] if title else "",
                "content": content[:5000] if content else "",  # 길이 제한
                "date": date[:20] if date else "",
                "url": url[:500] if url else "",
                "source": source[:200] if source else "",
                "category": category[:50] if category else "",
                "topic": topic[:100] if topic else "",
                "filename": filename[:200],
                "created_at": current_timestamp
            }
            
            milvus_data.append(data_item)
        
        logger.info(f"Prepared {len(milvus_data)} items for Milvus")
        return milvus_data

    def insert_to_milvus(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> bool:
        """
        Milvus 컬렉션에 데이터 삽입 (MilvusClient API 사용)
        """
        if not data:
            logger.warning("No data to insert")
            return False
        
        try:
            logger.info(f"Inserting {len(data)} items to collection '{collection_name}'...")
            
            # 배치 삽입
            for i in tqdm(range(0, len(data), batch_size), desc="Inserting batches"):
                batch_data = data[i:i+batch_size]
                
                # MilvusClient.insert()는 딕셔너리 리스트를 받습니다!
                res = self.client.insert(
                    collection_name=collection_name,
                    data=batch_data
                )
                
                logger.debug(f"Inserted batch {i//batch_size + 1}: {res}")
            
            logger.info(f"✅ Successfully inserted {len(data)} items to '{collection_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert data to Milvus: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_single_file(
        self,
        input_file: str,
        collection_name: str,
        use_gpt_metadata: bool = True
    ) -> bool:
        """단일 파일을 처리하여 Milvus에 저장"""
        start_time = time.time()
        filename = Path(input_file).name
        
        # 뉴스 데이터 로드
        news_data = self.load_news_data(input_file)
        if not news_data:
            logger.error(f"No valid news data found in {input_file}")
            return False
        
        self.stats['total_documents'] += len(news_data)
        
        # 컬렉션 생성
        self.create_collection(collection_name)
        
        # Milvus 데이터 준비
        milvus_data = self.prepare_milvus_data(news_data, filename, use_gpt_metadata)
        if not milvus_data:
            logger.error(f"No valid data prepared from {input_file}")
            return False
        
        self.stats['processed_documents'] += len(milvus_data)
        
        # 데이터 삽입
        success = self.insert_to_milvus(collection_name, milvus_data)
        
        processing_time = time.time() - start_time
        self.stats['processing_time'] += processing_time
        
        logger.info(f"✅ Processed {filename}: {len(milvus_data)} documents in {processing_time:.2f}s")
        
        # 컬렉션 통계 출력
        stats = self.client.get_collection_stats(collection_name)
        logger.info(f"Collection '{collection_name}' stats: {stats}")
        
        return success

    def print_summary(self):
        """변환 결과 요약 출력"""
        print("\n" + "="*60)
        print("🎉 Milvus Lite Conversion Summary (V2)")
        print("="*60)
        print(f"📊 Total Documents: {self.stats['total_documents']}")
        print(f"✅ Processed Documents: {self.stats['processed_documents']}")
        print(f"🗂️ Created Collections: {len(self.stats['created_collections'])}")
        print(f"⏱️ Processing Time: {self.stats['processing_time']:.2f} seconds")
        print(f"💾 Database File: {self.db_file}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Convert news data to Milvus Lite V2 (Official API)')
    parser.add_argument('--input', '-i', required=True,
                       help='Input file path')
    parser.add_argument('--collection', '-c', required=True,
                       help='Milvus collection name')
    parser.add_argument('--db-file', default='./milvus_lite_v2.db',
                       help='Milvus Lite database file path')
    parser.add_argument('--openai-key',
                       help='OpenAI API key for metadata extraction')
    parser.add_argument('--no-gpt', action='store_true',
                       help='Disable GPT-based metadata extraction')
    
    args = parser.parse_args()
    
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    
    if not openai_key and not args.no_gpt:
        logger.warning("No OpenAI API key provided. Using filename-based metadata extraction only.")
    
    converter = NewsToMilvusConverterV2(
        db_file=args.db_file,
        openai_api_key=openai_key
    )
    
    input_path = Path(args.input)
    use_gpt = not args.no_gpt and openai_key is not None
    
    if input_path.is_file():
        success = converter.process_single_file(str(input_path), args.collection, use_gpt)
    else:
        logger.error("Invalid input path")
        sys.exit(1)
    
    if success:
        converter.print_summary()
    else:
        logger.error("Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

