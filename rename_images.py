#!/usr/bin/env python3
"""
이미지 파일명을 fire1.jpg ~ fireN.jpg 형식으로 일괄 변경

사용법:
    python rename_images.py --input image_data/fire
    python rename_images.py --input image_data/fire --prefix fire --start 1
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import shutil

def get_image_files(directory: str) -> List[Path]:
    """
    디렉토리에서 이미지 파일들을 찾아서 정렬된 리스트로 반환
    
    Args:
        directory: 이미지 디렉토리 경로
    
    Returns:
        정렬된 이미지 파일 경로 리스트
    """
    image_dir = Path(directory)
    if not image_dir.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    
    # 이미지 파일 찾기
    image_files = []
    for file_path in image_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    # 파일명 기준 정렬
    image_files.sort(key=lambda x: x.name.lower())
    
    return image_files

def rename_images(
    directory: str,
    prefix: str = "fire",
    start_num: int = 1,
    extension: str = ".jpg",
    dry_run: bool = False,
    backup: bool = True
) -> List[Tuple[str, str]]:
    """
    이미지 파일들을 순차적으로 리네이밍
    
    Args:
        directory: 이미지 디렉토리
        prefix: 파일명 접두사 (기본값: "fire")
        start_num: 시작 번호 (기본값: 1)
        extension: 변경할 확장자 (기본값: ".jpg")
        dry_run: True면 실제 변경 없이 미리보기만
        backup: True면 백업 폴더 생성
    
    Returns:
        [(원본 파일명, 새 파일명), ...] 리스트
    """
    image_dir = Path(directory)
    image_files = get_image_files(directory)
    
    if not image_files:
        print(f"⚠️  No image files found in {directory}")
        return []
    
    print(f"\n📁 Found {len(image_files)} image files in {directory}")
    print(f"🏷️  Renaming pattern: {prefix}{{N}}{extension} (starting from {start_num})")
    print()
    
    # 백업 폴더 생성
    if backup and not dry_run:
        backup_dir = image_dir / "_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"💾 Backup folder created: {backup_dir}")
    
    changes = []
    current_num = start_num
    
    for img_file in image_files:
        # 새 파일명 생성
        new_name = f"{prefix}{current_num}{extension}"
        new_path = image_dir / new_name
        
        # 이미 올바른 이름이면 스킵
        if img_file.name == new_name:
            print(f"⏭️  Skip: {img_file.name} (already correct)")
            current_num += 1
            continue
        
        # 새 파일명이 이미 존재하는지 확인
        if new_path.exists() and new_path != img_file:
            print(f"⚠️  Warning: {new_name} already exists, skipping {img_file.name}")
            continue
        
        changes.append((img_file.name, new_name))
        
        if dry_run:
            print(f"🔍 [DRY RUN] {img_file.name:40s} → {new_name}")
        else:
            # 백업
            if backup:
                backup_path = backup_dir / img_file.name
                shutil.copy2(img_file, backup_path)
            
            # 리네임
            img_file.rename(new_path)
            print(f"✅ {img_file.name:40s} → {new_name}")
        
        current_num += 1
    
    return changes

def main():
    parser = argparse.ArgumentParser(
        description='Rename image files to sequential format (e.g., fire1.jpg, fire2.jpg, ...)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 사용 (fire1.jpg부터 시작)
  python rename_images.py --input image_data/fire
  
  # 미리보기 (실제 변경 없음)
  python rename_images.py --input image_data/fire --dry-run
  
  # 커스텀 접두사와 시작 번호
  python rename_images.py --input image_data/fire --prefix disaster --start 10
  
  # PNG 확장자로 변경
  python rename_images.py --input image_data/fire --ext .png
  
  # 백업 없이 실행
  python rename_images.py --input image_data/fire --no-backup
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing images')
    parser.add_argument('--prefix', '-p', default='fire',
                       help='Filename prefix (default: fire)')
    parser.add_argument('--start', '-s', type=int, default=1,
                       help='Starting number (default: 1)')
    parser.add_argument('--ext', '-e', default='.jpg',
                       help='Target file extension (default: .jpg)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Preview changes without actually renaming')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup folder')
    
    args = parser.parse_args()
    
    # 확장자 처리
    extension = args.ext if args.ext.startswith('.') else f'.{args.ext}'
    
    print("\n" + "="*70)
    print("🖼️  Image Renaming Tool")
    print("="*70)
    
    try:
        changes = rename_images(
            directory=args.input,
            prefix=args.prefix,
            start_num=args.start,
            extension=extension,
            dry_run=args.dry_run,
            backup=not args.no_backup
        )
        
        print("\n" + "="*70)
        if args.dry_run:
            print(f"🔍 [DRY RUN] Would rename {len(changes)} files")
            print("💡 Run without --dry-run to apply changes")
        else:
            print(f"✅ Successfully renamed {len(changes)} files")
            if not args.no_backup:
                print(f"💾 Original files backed up to: {args.input}/_backup")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

