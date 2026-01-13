#!/usr/bin/env python3
"""
Text-only DB Builder - 텍스트 전용 Milvus 컬렉션 생성

Usage:
    python build_text_db.py --input data/manuals/disaster_manuals.json --collection fire_manual --db-file db/fire.db
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from pymilvus import MilvusClient

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDBBuilder:
    def __init__(self, db_file: str, text_model: str = "jhgan/ko-sroberta-multitask"):
        self.db_file = db_file
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(db_file)
        logger.info(f"✅ Connected to {db_file}")
        
        device = 'cuda' if self._check_cuda() else 'cpu'
        self.text_embeddings = HuggingFaceEmbeddings(model_name=text_model, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True})
        logger.info(f"✅ Text model loaded on {device}")
    
    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def create_collection(self, collection_name: str, dimension: int = 768):
        if self.client.has_collection(collection_name):
            logger.info(f"Dropping existing collection '{collection_name}'")
            self.client.drop_collection(collection_name)
        self.client.create_collection(collection_name=collection_name, dimension=dimension, metric_type="COSINE")
        logger.info(f"✅ Created collection '{collection_name}' (dim={dimension})")
    
    def build_from_json(self, json_file: str, collection_name: str, batch_size: int = 50):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {json_file}")
        
        self.create_collection(collection_name)
        
        texts = [item.get('text', item.get('content', '')) for item in data]
        logger.info("Generating embeddings...")
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            embeddings = self.text_embeddings.embed_documents(batch)
            all_embeddings.extend(embeddings)
        
        milvus_data = []
        for idx, (item, embedding) in enumerate(zip(data, all_embeddings)):
            milvus_data.append({
                "id": idx,
                "vector": embedding,
                "doc_id": item.get('doc_id', f'doc_{idx}')[:200],
                "title": item.get('title', '')[:500],
                "content": item.get('text', item.get('content', ''))[:5000],
                "date": item.get('date', '')[:20],
                "url": item.get('url', '')[:500],
                "source": item.get('source', '')[:200],
                "category": item.get('category', '')[:50],
                "topic": item.get('topic', '')[:100],
                "has_image": False,
                "image_path": "",
                "image_embedding": "[]"
            })
        
        logger.info(f"Inserting {len(milvus_data)} items...")
        for i in tqdm(range(0, len(milvus_data), 1000), desc="Inserting"):
            self.client.insert(collection_name=collection_name, data=milvus_data[i:i+1000])
        
        logger.info(f"✅ Successfully built collection '{collection_name}' with {len(milvus_data)} items")
        return len(milvus_data)


def main():
    parser = argparse.ArgumentParser(description='Build text-only Milvus collection')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--collection', '-c', required=True, help='Collection name')
    parser.add_argument('--db-file', default='./db/fire.db', help='Milvus database file')
    parser.add_argument('--text-model', default='jhgan/ko-sroberta-multitask', help='Text embedding model')
    args = parser.parse_args()
    
    print(f"\n{'='*60}\n📦 Text DB Builder\n{'='*60}")
    print(f"Input:      {args.input}")
    print(f"Collection: {args.collection}")
    print(f"Database:   {args.db_file}")
    print("="*60 + "\n")
    
    builder = TextDBBuilder(db_file=args.db_file, text_model=args.text_model)
    count = builder.build_from_json(args.input, args.collection)
    
    print(f"\n✅ Done! {count} items indexed in '{args.collection}'")
    print(f"\n💡 Search example:")
    print(f"   python run_search.py --mode text -q '화재 대피' -c {args.collection} --db-file {args.db_file}")
    return 0


if __name__ == "__main__":
    exit(main())
