#!/usr/bin/env python3
"""
무료 이미지 다운로드 도구

무료 API를 사용하여 뉴스 토픽에 맞는 이미지 자동 다운로드

지원 소스:
- Unsplash (무료, 50 req/hour, API 키 필요)
- Pexels (무료, 200 req/hour, API 키 필요)
- Pixabay (무료, 100 req/min, API 키 필요)

사용법:
    # Unsplash
    python download_free_images.py \
      --source unsplash \
      --api-key YOUR_KEY \
      --query "fire disaster" \
      --output image_data/fire_downloaded \
      --limit 20

    # Pexels
    python download_free_images.py \
      --source pexels \
      --api-key YOUR_KEY \
      --query "fire emergency" \
      --output image_data/fire_downloaded \
      --limit 20
"""

import argparse
import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FreeImageDownloader:
    """무료 이미지 다운로드"""
    
    def __init__(self, source: str, api_key: str):
        self.source = source
        self.api_key = api_key
        self.session = requests.Session()
    
    def search_unsplash(self, query: str, per_page: int = 30) -> List[Dict[str, Any]]:
        """Unsplash API 검색"""
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {self.api_key}"}
        params = {"query": query, "per_page": per_page, "orientation": "landscape"}
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            images = []
            for item in data.get("results", []):
                images.append({
                    "id": item["id"],
                    "url": item["urls"]["regular"],  # 1080px width
                    "download_url": item["links"]["download"],
                    "photographer": item["user"]["name"],
                    "description": item.get("description", "")
                })
            
            logger.info(f"Found {len(images)} images on Unsplash")
            return images
        
        except Exception as e:
            logger.error(f"Unsplash API error: {e}")
            return []
    
    def search_pexels(self, query: str, per_page: int = 80) -> List[Dict[str, Any]]:
        """Pexels API 검색"""
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.api_key}
        params = {"query": query, "per_page": per_page, "orientation": "landscape"}
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            images = []
            for item in data.get("photos", []):
                images.append({
                    "id": str(item["id"]),
                    "url": item["src"]["large"],  # 1280px width
                    "photographer": item["photographer"],
                    "description": item.get("alt", "")
                })
            
            logger.info(f"Found {len(images)} images on Pexels")
            return images
        
        except Exception as e:
            logger.error(f"Pexels API error: {e}")
            return []
    
    def search_pixabay(self, query: str, per_page: int = 100) -> List[Dict[str, Any]]:
        """Pixabay API 검색"""
        url = "https://pixabay.com/api/"
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": per_page,
            "image_type": "photo",
            "orientation": "horizontal"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            images = []
            for item in data.get("hits", []):
                images.append({
                    "id": str(item["id"]),
                    "url": item["largeImageURL"],  # 1280px width
                    "photographer": item["user"],
                    "description": item.get("tags", "")
                })
            
            logger.info(f"Found {len(images)} images on Pixabay")
            return images
        
        except Exception as e:
            logger.error(f"Pixabay API error: {e}")
            return []
    
    def search_images(self, query: str, limit: int = 30) -> List[Dict[str, Any]]:
        """이미지 검색 (소스별)"""
        if self.source == "unsplash":
            return self.search_unsplash(query, per_page=min(limit, 30))
        elif self.source == "pexels":
            return self.search_pexels(query, per_page=min(limit, 80))
        elif self.source == "pixabay":
            return self.search_pixabay(query, per_page=min(limit, 100))
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    def download_image(self, image_info: Dict[str, Any], output_dir: Path, index: int) -> bool:
        """이미지 다운로드"""
        try:
            url = image_info["url"]
            image_id = image_info["id"]
            
            # 파일명 생성
            filename = f"{self.source}_{image_id}_{index:03d}.jpg"
            output_path = output_dir / filename
            
            # 다운로드
            response = requests.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filename}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to download image {image_info.get('id')}: {e}")
            return False
    
    def download_images(self, query: str, output_dir: str, limit: int = 30, delay: float = 0.5):
        """여러 이미지 다운로드"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 검색
        logger.info(f"Searching for '{query}' on {self.source}...")
        images = self.search_images(query, limit=limit)
        
        if not images:
            logger.error("No images found")
            return
        
        # 다운로드
        logger.info(f"Downloading {min(len(images), limit)} images...")
        downloaded = 0
        
        for i, image_info in enumerate(images[:limit]):
            if self.download_image(image_info, output_path, i):
                downloaded += 1
            
            # Rate limit 준수
            if i < len(images) - 1:
                time.sleep(delay)
        
        logger.info(f"✅ Downloaded {downloaded}/{limit} images to {output_dir}")


def print_api_setup_guide():
    """API 키 설정 가이드 출력"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         무료 이미지 API 키 받는 방법                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

📸 Unsplash (추천!)
   ├─ URL: https://unsplash.com/developers
   ├─ 제한: 50 requests/hour
   ├─ 품질: ⭐⭐⭐⭐⭐ (최고 품질, 고해상도)
   └─ 가입 → Create Application → Access Key 복사

🎨 Pexels
   ├─ URL: https://www.pexels.com/api/
   ├─ 제한: 200 requests/hour
   ├─ 품질: ⭐⭐⭐⭐ (고품질)
   └─ 가입 → API Key 받기

🖼️ Pixabay
   ├─ URL: https://pixabay.com/api/docs/
   ├─ 제한: 100 requests/minute
   ├─ 품질: ⭐⭐⭐ (양질)
   └─ 가입 → API Key 받기

사용 예시:
  export UNSPLASH_KEY="your_unsplash_access_key"
  
  python download_free_images.py \\
    --source unsplash \\
    --api-key $UNSPLASH_KEY \\
    --query "fire disaster emergency" \\
    --output image_data/fire_downloaded \\
    --limit 20

""")


def main():
    parser = argparse.ArgumentParser(
        description='Download free images from various sources',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--source', choices=['unsplash', 'pexels', 'pixabay'],
                       help='Image source')
    parser.add_argument('--api-key', help='API key for the selected source')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--output', default='downloaded_images',
                       help='Output directory')
    parser.add_argument('--limit', type=int, default=20,
                       help='Number of images to download')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between downloads (seconds)')
    parser.add_argument('--help-setup', action='store_true',
                       help='Show API setup guide')
    
    args = parser.parse_args()
    
    # Setup 가이드 출력
    if args.help_setup or not args.source:
        print_api_setup_guide()
        if not args.source:
            return
    
    # 필수 인자 확인
    if not args.api_key:
        print("❌ Error: --api-key is required")
        print("💡 Get your API key first (use --help-setup for guide)")
        return
    
    if not args.query:
        print("❌ Error: --query is required")
        return
    
    # 다운로드 실행
    print("\n" + "="*80)
    print(f"📥 Downloading images from {args.source.upper()}")
    print("="*80)
    print(f"Query: {args.query}")
    print(f"Output: {args.output}")
    print(f"Limit: {args.limit}")
    print("="*80 + "\n")
    
    downloader = FreeImageDownloader(args.source, args.api_key)
    downloader.download_images(args.query, args.output, limit=args.limit, delay=args.delay)
    
    print("\n" + "="*80)
    print("✅ Download complete!")
    print("="*80)
    print("\n💡 Next step: Match these images to your news")
    print(f"   python smart_image_matcher.py \\")
    print(f"     --news your_news.json \\")
    print(f"     --images {args.output} \\")
    print(f"     --output smart_mappings.json")
    print("")


if __name__ == "__main__":
    main()

