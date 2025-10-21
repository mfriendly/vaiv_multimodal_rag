#!/usr/bin/env python3
"""
Multimodal RAG 데모 - 화재 뉴스 + 이미지

이 스크립트는 다음을 수행합니다:
1. 화재 뉴스 데이터 로드
2. image_data/fire/ 폴더의 이미지를 임의로 할당
3. Multimodal 컬렉션 생성 및 데이터 삽입
4. 다양한 검색 방법 데모
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fire_news(news_file: str, limit: int = 100) -> List[Dict[str, Any]]:
    """화재 뉴스 데이터 로드"""
    logger.info(f"Loading news from {news_file}")
    
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
                    for doc in item["search_result"]:
                        news_list.append({
                            "doc_id": doc.get("doc_id", ""),
                            "title": doc.get("title", ""),
                            "text": doc.get("text", ""),
                            "date": doc.get("date", ""),
                            "url": doc.get("url", ""),
                            "source": doc.get("source", ""),
                            "category": "disaster",
                            "topic": "fire"
                        })
        else:
            # 이미 표준화된 구조
            news_list = raw_data
    
    # Limit 적용 및 필터링
    filtered_news = []
    for idx, news in enumerate(news_list[:limit]):
        if len(news.get("text", "")) > 50:  # 최소 길이
            news["doc_id"] = news.get("doc_id") or f"fire_news_{idx}"
            filtered_news.append(news)
    
    logger.info(f"Loaded {len(filtered_news)} news articles")
    return filtered_news


def assign_images_randomly(
    news_list: List[Dict[str, Any]], 
    image_dir: str,
    assign_ratio: float = 0.3
) -> Dict[str, str]:
    """
    이미지를 뉴스에 임의로 할당
    
    Args:
        news_list: 뉴스 리스트
        image_dir: 이미지 디렉토리
        assign_ratio: 이미지를 할당할 뉴스 비율 (0.0 ~ 1.0)
    
    Returns:
        {doc_id: image_path} 매핑
    """
    logger.info(f"Assigning images from {image_dir}")
    
    # 이미지 파일 찾기
    image_files = []
    image_path = Path(image_dir)
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(image_path.glob(ext)))
    
    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return {}
    
    logger.info(f"Found {len(image_files)} images")
    
    # 임의로 뉴스 선택
    num_to_assign = int(len(news_list) * assign_ratio)
    selected_news = random.sample(news_list, min(num_to_assign, len(news_list)))
    
    # 이미지 할당
    image_mappings = {}
    for news in selected_news:
        # 이미지 랜덤 선택 (중복 허용)
        image_file = random.choice(image_files)
        image_mappings[news["doc_id"]] = str(image_file)
    
    logger.info(f"Assigned {len(image_mappings)} images to news articles")
    return image_mappings


def save_mappings(image_mappings: Dict[str, str], output_file: str):
    """이미지 매핑을 JSON 파일로 저장"""
    mapping_list = [
        {"doc_id": doc_id, "image_path": img_path}
        for doc_id, img_path in image_mappings.items()
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved image mappings to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Multimodal RAG Demo with Fire News')
    parser.add_argument('--news', default='news_data/01_disaster_Fire_3years.json',
                       help='News data JSON file')
    parser.add_argument('--images', default='image_data/fire',
                       help='Image directory')
    parser.add_argument('--limit', type=int, default=100,
                       help='Number of news to process')
    parser.add_argument('--ratio', type=float, default=0.3,
                       help='Ratio of news to assign images (0.0 ~ 1.0)')
    parser.add_argument('--output-news', default='prepared_fire_news.json',
                       help='Output news JSON file')
    parser.add_argument('--output-images', default='fire_image_mappings.json',
                       help='Output image mappings JSON file')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔥 Multimodal RAG Demo - Fire News + Images")
    print("="*80)
    
    # 1. 뉴스 로드
    print("\n📰 Step 1: Loading fire news...")
    news_list = load_fire_news(args.news, limit=args.limit)
    
    if not news_list:
        logger.error("No news data loaded!")
        return
    
    print(f"   ✅ Loaded {len(news_list)} news articles")
    print(f"   📝 Sample title: {news_list[0]['title'][:50]}...")
    
    # 2. 이미지 할당
    print("\n🖼️  Step 2: Assigning images randomly...")
    image_mappings = assign_images_randomly(news_list, args.images, assign_ratio=args.ratio)
    
    if image_mappings:
        print(f"   ✅ Assigned {len(image_mappings)} images")
        sample_doc_id = list(image_mappings.keys())[0]
        print(f"   📷 Sample: {sample_doc_id} -> {Path(image_mappings[sample_doc_id]).name}")
    else:
        print(f"   ⚠️  No images assigned (check {args.images})")
    
    # 3. 파일 저장
    print("\n💾 Step 3: Saving prepared data...")
    
    # 뉴스 저장
    with open(args.output_news, 'w', encoding='utf-8') as f:
        json.dump(news_list, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved news to: {args.output_news}")
    
    # 이미지 매핑 저장
    if image_mappings:
        save_mappings(image_mappings, args.output_images)
        print(f"   ✅ Saved image mappings to: {args.output_images}")
    
    # 4. 통계
    print("\n📊 Statistics:")
    print(f"   Total news: {len(news_list)}")
    print(f"   News with images: {len(image_mappings)}")
    print(f"   News without images: {len(news_list) - len(image_mappings)}")
    print(f"   Image assignment ratio: {len(image_mappings)/len(news_list)*100:.1f}%")
    
    # 5. 다음 단계 안내
    print("\n" + "="*80)
    print("✅ Data preparation complete!")
    print("="*80)
    print("\n🚀 Next steps:")
    print(f"\n1. Create multimodal collection and insert data:")
    print(f"   python multimodal_rag_v2.py \\")
    print(f"     --mode create \\")
    print(f"     --collection fire_multimodal \\")
    print(f"     --input {args.output_news} \\")
    print(f"     --images {args.output_images}")
    
    print(f"\n2. Search by text:")
    print(f"   python multimodal_rag_v2.py \\")
    print(f"     --mode search \\")
    print(f"     --collection fire_multimodal \\")
    print(f"     --query '화재 사건'")
    
    print(f"\n3. Search by image:")
    print(f"   python multimodal_rag_v2.py \\")
    print(f"     --mode search-image \\")
    print(f"     --collection fire_multimodal \\")
    print(f"     --image {args.images}/fire1.jpg")
    
    print(f"\n4. Hybrid search (text + image):")
    print(f"   python multimodal_rag_v2.py \\")
    print(f"     --mode hybrid \\")
    print(f"     --collection fire_multimodal \\")
    print(f"     --query '대형 화재' \\")
    print(f"     --image {args.images}/fire2.jpg")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

