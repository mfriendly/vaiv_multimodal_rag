#!/usr/bin/env python3
"""
뉴스-이미지 수동 매칭 도구

특정 뉴스에 특정 이미지를 수동으로 매칭하여 JSON 파일로 저장합니다.

사용법:
    # 대화형 모드 (추천)
    python manual_image_matcher.py --news news_data/01_disaster_Fire_3years.json --images image_data/fire
    
    # CLI 모드 (직접 매칭)
    python manual_image_matcher.py --add "fire_news_001:fire1.jpg" --add "fire_news_002:fire3.jpg" --output mappings.json
    
    # 기존 매핑 파일 수정
    python manual_image_matcher.py --edit mappings.json --news news_data/01_disaster_Fire_3years.json --images image_data/fire
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

class ManualImageMatcher:
    """뉴스-이미지 수동 매칭 도구"""
    
    def __init__(self, news_file: Optional[str] = None, image_dir: Optional[str] = None):
        self.news_list = []
        self.image_files = []
        self.mappings = []
        
        if news_file:
            self.load_news(news_file)
        if image_dir:
            self.load_images(image_dir)
    
    def load_news(self, news_file: str):
        """뉴스 데이터 로드"""
        print(f"\n📰 Loading news from {news_file}...")
        
        with open(news_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 데이터 구조 파싱
        if isinstance(raw_data, list) and len(raw_data) > 0:
            first_item = raw_data[0]
            
            if "item" in first_item and "documentList" in first_item["item"]:
                # item.documentList 구조
                for item in raw_data:
                    if "item" in item and "documentList" in item["item"]:
                        for doc in item["item"]["documentList"]:
                            self.news_list.append({
                                "doc_id": doc.get("docID", ""),
                                "title": doc.get("title", ""),
                                "text": doc.get("content", ""),
                                "date": doc.get("date", "")
                            })
            elif "search_result" in first_item:
                # search_result 구조
                for item in raw_data:
                    if "search_result" in item:
                        self.news_list.extend(item["search_result"])
            else:
                # 이미 표준화된 구조
                self.news_list = raw_data
        
        print(f"✅ Loaded {len(self.news_list)} news articles")
    
    def load_images(self, image_dir: str):
        """이미지 파일 로드"""
        print(f"\n🖼️  Loading images from {image_dir}...")
        
        image_path = Path(image_dir)
        if not image_path.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        # 이미지 파일 찾기
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
        for file_path in sorted(image_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                self.image_files.append(str(file_path))
        
        print(f"✅ Found {len(self.image_files)} images")
    
    def load_mappings(self, mapping_file: str):
        """기존 매핑 파일 로드"""
        print(f"\n📂 Loading existing mappings from {mapping_file}...")
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
        
        print(f"✅ Loaded {len(self.mappings)} existing mappings")
    
    def add_mapping(self, doc_id: str, image_path: str, caption: str = ""):
        """매핑 추가"""
        # 기존 매핑 제거 (같은 doc_id)
        self.mappings = [m for m in self.mappings if m.get("doc_id") != doc_id]
        
        # 새 매핑 추가
        mapping = {
            "doc_id": doc_id,
            "image_path": image_path
        }
        if caption:
            mapping["caption"] = caption
        
        self.mappings.append(mapping)
        print(f"✅ Added mapping: {doc_id} → {Path(image_path).name}")
    
    def remove_mapping(self, doc_id: str):
        """매핑 제거"""
        original_len = len(self.mappings)
        self.mappings = [m for m in self.mappings if m.get("doc_id") != doc_id]
        
        if len(self.mappings) < original_len:
            print(f"✅ Removed mapping for: {doc_id}")
        else:
            print(f"⚠️  No mapping found for: {doc_id}")
    
    def save_mappings(self, output_file: str):
        """매핑을 JSON 파일로 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.mappings, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Saved {len(self.mappings)} mappings to {output_file}")
    
    def show_news(self, limit: int = 10, search: str = ""):
        """뉴스 목록 표시"""
        print("\n" + "="*80)
        print("📰 News List")
        print("="*80)
        
        filtered_news = self.news_list
        if search:
            filtered_news = [
                n for n in self.news_list 
                if search.lower() in n.get("title", "").lower() or 
                   search.lower() in n.get("doc_id", "").lower()
            ]
        
        for idx, news in enumerate(filtered_news[:limit]):
            doc_id = news.get("doc_id", "unknown")
            title = news.get("title", "No title")[:60]
            date = news.get("date", "")
            
            # 이미 매핑되어 있는지 확인
            existing = next((m for m in self.mappings if m["doc_id"] == doc_id), None)
            status = "✅" if existing else "  "
            
            print(f"{status} [{idx+1:3d}] {doc_id:30s} | {title:60s} | {date}")
        
        if len(filtered_news) > limit:
            print(f"\n... and {len(filtered_news) - limit} more (use --limit to see more)")
    
    def show_images(self):
        """이미지 목록 표시"""
        print("\n" + "="*80)
        print("🖼️  Image List")
        print("="*80)
        
        for idx, img_path in enumerate(self.image_files):
            img_name = Path(img_path).name
            
            # 이미 매핑되어 있는지 확인
            usage_count = sum(1 for m in self.mappings if m["image_path"] == img_path)
            status = f"({usage_count})" if usage_count > 0 else "  "
            
            print(f"{status} [{idx+1:3d}] {img_name}")
    
    def show_mappings(self):
        """현재 매핑 목록 표시"""
        print("\n" + "="*80)
        print("🔗 Current Mappings")
        print("="*80)
        
        if not self.mappings:
            print("No mappings yet.")
            return
        
        for idx, mapping in enumerate(self.mappings):
            doc_id = mapping.get("doc_id", "unknown")
            img_path = mapping.get("image_path", "")
            img_name = Path(img_path).name
            caption = mapping.get("caption", "")
            
            # 뉴스 제목 찾기
            news = next((n for n in self.news_list if n.get("doc_id") == doc_id), None)
            title = news.get("title", "Unknown")[:50] if news else "Unknown"
            
            caption_str = f" | {caption}" if caption else ""
            print(f"[{idx+1:3d}] {doc_id:30s} → {img_name:20s} | {title}{caption_str}")
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n" + "="*80)
        print("🎯 Interactive Mapping Mode")
        print("="*80)
        print("\nCommands:")
        print("  n [search]   - Show news list (optional: search term)")
        print("  i            - Show image list")
        print("  m            - Show current mappings")
        print("  a            - Add new mapping")
        print("  r            - Remove mapping")
        print("  s <file>     - Save to file")
        print("  q            - Quit")
        print("="*80)
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if not command:
                    continue
                
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if cmd == 'q' or cmd == 'quit':
                    print("👋 Goodbye!")
                    break
                
                elif cmd == 'n' or cmd == 'news':
                    self.show_news(limit=20, search=arg)
                
                elif cmd == 'i' or cmd == 'images':
                    self.show_images()
                
                elif cmd == 'm' or cmd == 'mappings':
                    self.show_mappings()
                
                elif cmd == 'a' or cmd == 'add':
                    self._interactive_add()
                
                elif cmd == 'r' or cmd == 'remove':
                    self._interactive_remove()
                
                elif cmd == 's' or cmd == 'save':
                    filename = arg if arg else input("Output filename: ").strip()
                    if filename:
                        self.save_mappings(filename)
                    else:
                        print("❌ Filename required")
                
                elif cmd == 'h' or cmd == 'help':
                    print("\nCommands:")
                    print("  n [search]   - Show news list")
                    print("  i            - Show image list")
                    print("  m            - Show current mappings")
                    print("  a            - Add new mapping")
                    print("  r            - Remove mapping")
                    print("  s <file>     - Save to file")
                    print("  q            - Quit")
                
                else:
                    print(f"❌ Unknown command: {cmd} (type 'h' for help)")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _interactive_add(self):
        """대화형 매핑 추가"""
        # 뉴스 선택
        doc_id = input("News doc_id (or index): ").strip()
        
        # 인덱스로 입력한 경우
        if doc_id.isdigit():
            idx = int(doc_id) - 1
            if 0 <= idx < len(self.news_list):
                doc_id = self.news_list[idx].get("doc_id", "")
            else:
                print(f"❌ Invalid index: {doc_id}")
                return
        
        # doc_id 확인
        news = next((n for n in self.news_list if n.get("doc_id") == doc_id), None)
        if not news:
            print(f"❌ News not found: {doc_id}")
            return
        
        print(f"📰 Selected: {news.get('title', '')[:60]}")
        
        # 이미지 선택
        img_input = input("Image filename or index: ").strip()
        
        # 인덱스로 입력한 경우
        if img_input.isdigit():
            idx = int(img_input) - 1
            if 0 <= idx < len(self.image_files):
                image_path = self.image_files[idx]
            else:
                print(f"❌ Invalid index: {img_input}")
                return
        else:
            # 파일명으로 찾기
            matches = [f for f in self.image_files if img_input in f]
            if len(matches) == 1:
                image_path = matches[0]
            elif len(matches) > 1:
                print(f"❌ Multiple matches found: {[Path(m).name for m in matches]}")
                return
            else:
                print(f"❌ Image not found: {img_input}")
                return
        
        print(f"🖼️  Selected: {Path(image_path).name}")
        
        # 캡션 (선택)
        caption = input("Caption (optional, press Enter to skip): ").strip()
        
        # 매핑 추가
        self.add_mapping(doc_id, image_path, caption)
    
    def _interactive_remove(self):
        """대화형 매핑 제거"""
        doc_id = input("News doc_id to remove: ").strip()
        self.remove_mapping(doc_id)


def main():
    parser = argparse.ArgumentParser(
        description='Manual news-image matching tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python manual_image_matcher.py --news news_data/01_disaster_Fire_3years.json --images image_data/fire
  
  # Add mappings directly
  python manual_image_matcher.py --add "fire_news_001:fire1.jpg" --add "fire_news_002:fire3.jpg" --output mappings.json
  
  # Edit existing mappings
  python manual_image_matcher.py --edit mappings.json --news news_data/01_disaster_Fire_3years.json --images image_data/fire
  
  # Add to existing mappings
  python manual_image_matcher.py --edit mappings.json --add "fire_news_003:fire5.jpg" --output mappings_updated.json
        """
    )
    
    parser.add_argument('--news', '-n',
                       help='News JSON file')
    parser.add_argument('--images', '-i',
                       help='Image directory')
    parser.add_argument('--edit', '-e',
                       help='Edit existing mapping file')
    parser.add_argument('--add', '-a', action='append',
                       help='Add mapping in format "doc_id:image_filename" (can be used multiple times)')
    parser.add_argument('--remove', '-r', action='append',
                       help='Remove mapping by doc_id (can be used multiple times)')
    parser.add_argument('--output', '-o', default='manual_image_mappings.json',
                       help='Output JSON file (default: manual_image_mappings.json)')
    parser.add_argument('--interactive', action='store_true',
                       help='Force interactive mode')
    
    args = parser.parse_args()
    
    # Matcher 초기화
    matcher = ManualImageMatcher(args.news, args.images)
    
    # 기존 매핑 로드
    if args.edit:
        matcher.load_mappings(args.edit)
    
    # CLI 모드로 매핑 추가
    if args.add:
        for mapping_str in args.add:
            if ':' not in mapping_str:
                print(f"❌ Invalid format: {mapping_str} (expected doc_id:image_filename)")
                continue
            
            doc_id, img_filename = mapping_str.split(':', 1)
            
            # 이미지 경로 찾기
            if matcher.image_files:
                matches = [f for f in matcher.image_files if img_filename in f]
                if matches:
                    image_path = matches[0]
                else:
                    # 전체 경로로 시도
                    image_path = img_filename
            else:
                image_path = img_filename
            
            matcher.add_mapping(doc_id.strip(), image_path.strip())
    
    # CLI 모드로 매핑 제거
    if args.remove:
        for doc_id in args.remove:
            matcher.remove_mapping(doc_id.strip())
    
    # 대화형 모드 또는 자동 저장
    if args.interactive or (not args.add and not args.remove and not args.edit):
        # 대화형 모드
        if matcher.news_list and matcher.image_files:
            matcher.interactive_mode()
            
            # 저장 여부 확인
            save = input(f"\n💾 Save mappings to {args.output}? (y/n): ").strip().lower()
            if save == 'y' or save == 'yes':
                matcher.save_mappings(args.output)
        else:
            print("❌ Both --news and --images are required for interactive mode")
            return 1
    else:
        # CLI 모드 - 자동 저장
        matcher.save_mappings(args.output)
    
    return 0

if __name__ == "__main__":
    exit(main())

