import os
os.environ["HF_HOME"] = "/mnt/nvme02/home/tdrag/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/nvme02/home/tdrag/.cache/huggingface"

from transformers.utils import move_cache
move_cache()

from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
import logging, os, re
from pathlib import Path
import datetime
import warnings
import json
import torch

from utils import analyze_qa_type

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Transformers 관련 경고 억제
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import transformers
transformers.logging.set_verbosity_error()

# HuggingFace 관련 경고 억제
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# 추가 경고 억제 설정
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# CUDA 최적화 설정
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# GPU 메모리 관리 설정
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB 사용 가능")

BASE_RETRIEVER_MODEL = "Facebook/rag-sequence-nq" # basic retriever model

logging.basicConfig(
    filename=f'vectordb_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
    )
global_manager = None
global_retriever = None
global_generator_model = "OpenAI MCQ"  # 기본 생성기 모델
global_retriever_model = "Facebook/rag-sequence-nq"  # 기본 검색자 모델
global_tokenizer = None
global_api_key = None  # OpenAI API 키를 위한 전역 변수
global_hybrid_search = None  # 하이브리드 검색 엔진

import pickle
from pydantic import BaseModel
from typing import Dict, Any
from tqdm import tqdm

# load data from other file
from utils import load_news_data
from manager import VectorStoreManager
from search import SearchInterface

from retrieval.dpr import run_dpr_question, load_model
from retrieval.gcs import search as gcs_search, parse_article

# langchain imports
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import openai
import time

# utils.py
from utils import (
    load_news_data, create_documents, create_faiss_index, create_chunks, process_date_string, retrieve_single_question, compute_relative_date
)


# hybrid search
from hybrid_search import HybridSearchEngine, create_comprehensive_answer
from uuid import uuid4

from evaluate import accuracy, gen_eval

# keys.py
from keys import GCS_KEY, ENGINE_KEY, OPENAI_API_KEY, MODEL_PATH, MODEL_NAMES, EXTRACTOR_MODEL_PATH, COHERE_API_KEY

global_api_key = OPENAI_API_KEY  # OpenAI API 키 설정

global_extractor_model_path = EXTRACTOR_MODEL_PATH


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / f'vectordb_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def initialize_manager():
    """전역 manager 및 모델 초기화"""
    global global_manager, global_retriever, global_retriever_model, global_tokenizer, global_hybrid_search

    # GPU 사용 가능한지 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},  # GPU 사용 설정
        encode_kwargs={'normalize_embeddings': False}
    )

    # 검색 모델 미리 로드

    if global_retriever is None or global_retriever_model is None or global_tokenizer is None:
        print("Loading retrieval models...")
        retriever, retriever_model, tokenizer = load_model(
            BASE_RETRIEVER_MODEL,
            top_k=25,  # 최대값으로 설정하여 재로드 방지
            device=device
        )
        global_retriever = retriever
        global_retriever_model = retriever_model
        global_tokenizer = tokenizer
        print("Retrieval models loaded successfully")


    base_dir = Path("/mnt/nvme02/home/tdrag/vaiv/RTRAG/faiss_indexes_metadata")  # FAISS DB 경로 - 메타데이터 기반 인덱싱
    
    # 모든 FAISS 인덱스 디렉토리 찾기 (날짜, 카테고리, 토픽 포함)
    sub_dirs = []
    if base_dir.exists():
        for d in base_dir.iterdir():
            if d.is_dir():
                # 날짜별 인덱스 (YYYYMM 또는 YYYYMMDD 형식)
                if re.match(r"^(date_)?\d{6,8}$", d.name):
                    sub_dirs.append(d)
                # 병합된 날짜 인덱스
                elif re.match(r"merged_\d{4,8}", d.name):
                    sub_dirs.append(d)
                # 카테고리별 인덱스
                elif d.name.startswith("category_"):
                    sub_dirs.append(d)
                # 토픽별 인덱스
                elif d.name.startswith("topic_"):
                    sub_dirs.append(d)
                # 통합 인덱스 (새로운 효율적 방식)
                elif d.name.startswith("unified_"):
                    sub_dirs.append(d)

    if not sub_dirs:
        print(f"No FAISS index directories found in {base_dir}")
        print("Available directories:", [d.name for d in base_dir.iterdir() if d.is_dir()] if base_dir.exists() else "Base directory doesn't exist")
    else:
        print(f"Found {len(sub_dirs)} FAISS index directories:")
        for d in sub_dirs:
            index_type = "date" if re.match(r"^(date_)?\d{6,8}$", d.name) else \
                        "merged" if d.name.startswith("merged_") else \
                        "category" if d.name.startswith("category_") else \
                        "topic" if d.name.startswith("topic_") else \
                        "unified" if d.name.startswith("unified_") else "unknown"
            print(f"  - {d.name} ({index_type})")

    global_manager = VectorStoreManager(embeddings, base_dir)
    
    
    # 하이브리드 검색 엔진 초기화
    if global_hybrid_search is None:
        # Google API 키는 keys.py에서 기본으로 가져오기
        try:
            from keys import GOOGLE_API_KEY, GOOGLE_CSE_ID
            google_api_key = GOOGLE_API_KEY
            google_cse_id = GOOGLE_CSE_ID
            print("✅ Google API keys loaded from keys.py")
        except ImportError:
            # keys.py에 없으면 환경변수에서 가져오기
            google_api_key = os.getenv('GOOGLE_API_KEY')
            google_cse_id = os.getenv('GOOGLE_CSE_ID')
            print("⚠️ Google API keys loaded from environment variables")
        
        global_hybrid_search = HybridSearchEngine(
            vector_manager=global_manager,
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )

    # 모델 워밍업 실행
    warmup_models()

    return global_manager

# GPT를 사용하여 쿼리에서 메타데이터(날짜, 카테고리, 토픽) 추출
def return_date_info(query_input):
    import time
    from typing import List, Dict, Optional, Tuple
    start_time = time.time()

    # 결과 해석
    results_output = "Results from query:\n"
    results_output += f"Query: {query_input}\n\n"

    # 메타데이터 저장
    extracted_dates = []
    metadata = {
        "category": None,
        "topic": None,
        "date_range": None
    }

    try:
        import openai
        client = openai.OpenAI(api_key=global_api_key)

        # GPT를 사용하여 메타데이터 추출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a metadata extraction specialist. Extract metadata from the given query in JSON format with these fields:

                1. dates: Array of dates in YYYYMMDD format. If a date range is mentioned, include both start and end dates. If only one date is mentioned, include it twice. If no specific date is mentioned, return empty array.
                2. category: Main news category from this list: ['politics', 'economics', 'society', 'culture', 'technology', 'sports', 'entertainment', 'disaster', 'crime', 'health', 'environment', 'international', 'other']. Return null if unclear.
                3. topic: Specific topic or keyword (max 2 words) that best describes the main subject. Return null if unclear.

                Rules:
                - Always respond with valid JSON only
                - Use YYYYMMDD format for dates (e.g., 20240115)
                - Choose the most appropriate category from the provided list
                - Keep topics concise and specific

                Example response:
                {"dates": ["20220115", "20220201"], "category": "disaster", "topic": "heavy snow"}"""},
                {"role": "user", "content": "What are the major fire incidents in the past 3 years?"},
                {"role": "assistant", "content": '''{"dates": ["20210101", "20231231"], "category": "disaster", "topic": "fire"}'''},
                {"role": "user", "content": "COVID-19 infection trends in 2021-2022"},
                {"role": "assistant", "content": '''{"dates": ["20210101", "20221231"], "category": "disaster", "topic": "infection"}'''},
                {"role": "user", "content": "Recent earthquake news and damage reports"},
                {"role": "assistant", "content": '''{"dates": [], "category": "disaster", "topic": "earthquake"}'''},
                {"role": "user", "content": query_input}
            ],
            max_tokens=150,
            temperature=0.1
        )

        llm_results = response.choices[0].message.content.strip()
        try:
            parsed_metadata = json.loads(llm_results)

            # 날짜 정보 처리
            if parsed_metadata.get("dates"):
                extracted_dates.extend(parsed_metadata["dates"])
                results_output += f"Date Range Found: {parsed_metadata['dates'][0]} to {parsed_metadata['dates'][-1]}\n"

            # 카테고리와 토픽 정보 처리
            metadata["category"] = parsed_metadata.get("category")
            metadata["topic"] = parsed_metadata.get("topic")

            if metadata["category"] or metadata["topic"]:
                results_output += "\n### Category/Topic Information\n"
                if metadata["category"]:
                    results_output += f"Category: {metadata['category']}\n"
                if metadata["topic"]:
                    results_output += f"Topic: {metadata['topic']}\n"

        except json.JSONDecodeError:
            results_output += "Error parsing metadata from GPT response.\n"

    except Exception as e:
        results_output += f"\n### Error in Date Extraction\n\nError: {str(e)}\n"

    processing_time = time.time() - start_time

    # 날짜 정보가 없는 경우 기본값 설정 (뉴스 데이터 범위에 맞춤)
    if not extracted_dates:
        end_date = "20231231"  # 뉴스 데이터 최대 범위
        start_date = "20210101"  # 뉴스 데이터 시작 범위
        extracted_dates = [start_date, end_date]

    # 날짜 정보 정렬 및 시작/종료 날짜 설정
    extracted_dates.sort()
    date_range = f"{extracted_dates[0]}/{extracted_dates[-1]}"

    results_output += f"\n### Date Range for Search\n{date_range}\n"

    # 결과와 메타데이터 반환
    return results_output, date_range, metadata

# FAISS 변환 전용 함수
def convert_to_faiss_indexes(files, use_metadata=True, category_filter=None, topic_filter=None):
    """파일들을 FAISS 인덱스로 변환하는 전용 함수"""
    global global_manager, global_api_key
    if global_manager is None:
        global_manager = initialize_manager()

    if not files:
        return None, "Please upload at least one file."

    processed_data = {
        "Number_of_indexes": 0,
        "Index_by_date": {},
        "Index_by_category": {},
        "Index_by_topic": {},
        "Current_indices": {}
    }

    start_time = time.time()
    total_docs = 0
    progress_html = ""

    try:
        for file_idx, file in enumerate(files, 1):
            progress_html += f"<p>Converting File {file_idx}/{len(files)}: {file.name} to FAISS Index</p>"
            yield processed_data, progress_html

            news_data = load_news_data(file.name)
            
            # 뉴스 데이터 전처리
            if "search_result" in list(news_data[0].keys()):
                news_date_temp = [news["search_result"] for news in news_data]
                news_data = [item for sublist in news_date_temp for item in sublist]
            elif "item" in list(news_data[0].keys()):
                news_data_before = news_data[0]["item"]["documentList"]
                new_news_data = []
                for d in news_data_before:
                    news_object = {
                        "date": d["date"],
                        "title": d["title"],
                        "text": d["content"],
                        "doc_id": d['docID'],
                        "url": d['url'],
                        "source": d['writerName'],
                        "vks": d['vks']
                    }
                    new_news_data.append(news_object)
                news_data = new_news_data.copy()

            # 메타데이터 추출 및 분류
            categorized_docs = {}
            topic_docs = {}
            date_docs = {}

            progress_html += f"<p>Extracting metadata and categorizing documents...</p>"
            yield processed_data, progress_html

            for news_idx, news in enumerate(tqdm(news_data, desc="Processing and categorizing news data")):
                # 기본 문서 생성
                content = news.get('text', news.get('content', ''))
                title = news.get('title', 'No Title')
                news_date = news.get('date', '20000101')
                if isinstance(news_date, str):
                    news_date = process_date_string(news_date)
                
                # 카테고리 및 토픽 추출 (GPT 사용)
                if use_metadata and global_api_key:
                    category, topic = extract_category_topic(content, title, global_api_key)
                else:
                    category, topic = "general", "general"

                # 필터링 적용
                if category_filter and category.lower() != category_filter.lower():
                    continue
                if topic_filter and topic.lower() != topic_filter.lower():
                    continue

                # 문서 생성
                doc = Document(
                    page_content=f"Title: {title}\nContent: {content}",
                    metadata={
                        'title': title,
                        'doc_id': news.get('doc_id', str(uuid4())),
                        'date': news_date,
                        'source': news.get('source', 'Unknown'),
                        'category': category,
                        'topic': topic,
                        'url': news.get('url', '')
                    }
                )

                # 날짜별 분류
                if news_date not in date_docs:
                    date_docs[news_date] = []
                date_docs[news_date].append(doc)

                # 카테고리별 분류
                if category not in categorized_docs:
                    categorized_docs[category] = []
                categorized_docs[category].append(doc)

                # 토픽별 분류
                if topic not in topic_docs:
                    topic_docs[topic] = []
                topic_docs[topic].append(doc)

                total_docs += 1

            # 🚀 새로운 효율적 FAISS 인덱스 생성 방식
            all_docs = list(categorized_docs.values())[0] if categorized_docs else []
            if not all_docs:
                # 모든 문서를 하나의 리스트로 합치기
                all_docs = []
                for docs in date_docs.values():
                    all_docs.extend(docs)
            
            if all_docs:
                progress_html += f"<p>🚀 Creating unified FAISS index (efficient method)...</p>"
                yield processed_data, progress_html
                
                # 통합 인덱스 생성 (모든 문서 포함, 메타데이터로 필터링)
                file_basename = file.name.split('.')[0] if hasattr(file, 'name') else 'manual_upload'
                unified_index_name = f"unified_{file_basename}"
                
                global_manager.create_index(unified_index_name, all_docs)
                
                # 통계 업데이트
                processed_data["Index_by_unified"] = {unified_index_name: len(all_docs)}
                processed_data["Index_by_date"] = {date: len(docs) for date, docs in date_docs.items()}
                processed_data["Index_by_category"] = {cat: len(docs) for cat, docs in categorized_docs.items()}
                processed_data["Index_by_topic"] = {topic: len(docs) for topic, docs in topic_docs.items()}
                
                print(f"✅ Unified index created: {unified_index_name} with {len(all_docs)} documents")
                print(f"📊 Contains {len(date_docs)} date groups, {len(categorized_docs)} categories, {len(topic_docs)} topics")
                
                # 선택적으로 큰 데이터셋의 경우 날짜별 분할도 생성
                if len(all_docs) > 1000:
                    progress_html += f"<p>Creating additional date-based indexes for large dataset...</p>"
                    yield processed_data, progress_html
                    
                    for date, docs in date_docs.items():
                        if len(docs) > 50:  # 최소 문서 수 조건
                            index_name = f"date_{date}"
                            global_manager.create_index(index_name, docs)
                            print(f"✅ Additional date index created: {index_name} with {len(docs)} documents")
            else:
                progress_html += f"<p style='color: red;'>❌ No documents to create indexes</p>"
                yield processed_data, progress_html

        # 통계 계산 (통합 인덱스 포함)
        unified_count = len(processed_data.get("Index_by_unified", {}))
        date_count = len(processed_data.get("Index_by_date", {}))
        category_count = len(processed_data.get("Index_by_category", {}))
        topic_count = len(processed_data.get("Index_by_topic", {}))
        
        processed_data["Number_of_indexes"] = unified_count + date_count + category_count + topic_count
        
        time_spend = time.time() - start_time
        status_msg = f"✅ FAISS Conversion Completed (Efficient Method):\n{len(files)} files, {total_docs} documents processed.\n"
        if unified_count > 0:
            status_msg += f"🚀 Created {unified_count} unified index(es) (efficient storage)\n"
            status_msg += f"📊 Metadata coverage: {category_count} categories, {topic_count} topics across {date_count} date groups\n"
        else:
            status_msg += f"Created {date_count} date indexes, {category_count} category indexes, {topic_count} topic indexes.\n"
        status_msg += f"📁 Saved to: {global_manager.base_dir}\n"
        status_msg += f"Time spent: {time_spend:.2f} seconds"

        # 생성된 인덱스 폴더 확인
        created_folders = []
        if global_manager.base_dir.exists():
            for folder in global_manager.base_dir.iterdir():
                if folder.is_dir():
                    created_folders.append(folder.name)
        
        if created_folders:
            status_msg += f"\n📂 Created folders: {', '.join(created_folders)}"
        else:
            status_msg += f"\n⚠️ No folders found in {global_manager.base_dir}"

        progress_html += f"<p style='color: green;'>{status_msg}</p>"
        yield processed_data, progress_html

    except Exception as e:
        error_msg = f"❌ FAISS conversion error: {str(e)}"
        progress_html += f"<p style='color: red;'>{error_msg}</p>"
        yield None, progress_html

def get_news_data_files():
    """news_data 폴더의 JSON 파일 목록 반환"""
    news_data_dir = Path("/mnt/nvme02/home/tdrag/vaiv/RTRAG/news_data")
    if not news_data_dir.exists():
        return []
    
    json_files = []
    for file_path in news_data_dir.glob("*.json"):
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        file_info = f"{file_path.name} ({file_size:.0f}MB)"
        json_files.append((str(file_path), file_info))
    
    return sorted(json_files, key=lambda x: x[1])  # 이름순 정렬

def auto_convert_news_data_to_faiss(selected_files=None, use_metadata=True, 
                                   category_filter=None, topic_filter=None):
    """news_data 폴더의 파일들을 자동으로 FAISS로 변환"""
    global global_manager, global_api_key
    
    if global_manager is None:
        global_manager = initialize_manager()
    
    news_data_dir = Path("/mnt/nvme02/home/tdrag/vaiv/RTRAG/news_data")
    
    # 파일 선택 로직
    if selected_files:
        # 선택된 파일들만 처리
        files_to_process = [Path(f) for f in selected_files if Path(f).exists()]
    else:
        # 모든 JSON 파일 처리
        files_to_process = list(news_data_dir.glob("*.json"))
    
    if not files_to_process:
        yield None, "No files found to process."
        return
    
    processed_data = {
        "Number_of_indexes": 0,
        "Index_by_date": {},
        "Index_by_category": {},
        "Index_by_topic": {},
        "Current_indices": {},
        "Processed_files": []
    }
    
    start_time = time.time()
    total_docs = 0
    progress_html = f"<h3>🚀 Auto-converting {len(files_to_process)} files from news_data/</h3>"
    
    try:
        for file_idx, file_path in enumerate(files_to_process, 1):
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            progress_html += f"<p>📄 Processing File {file_idx}/{len(files_to_process)}: {file_path.name} ({file_size_mb:.0f}MB)</p>"
            yield processed_data, progress_html
            
            # 파일 타입에 따른 카테고리 자동 감지
            filename = file_path.name.lower()
            auto_category = "disaster"  # 기본값
            auto_topic = None
            
            # 파일명에서 토픽 추출
            if "fire" in filename:
                auto_topic = "fire"
            elif "crime" in filename:
                auto_category = "crime"
                auto_topic = "crime"
            elif "snow" in filename:
                auto_topic = "heavy snow"
            elif "earthquake" in filename:
                auto_topic = "earthquake"
            elif "infection" in filename:
                auto_topic = "infection"
            elif "traffic" in filename:
                auto_topic = "traffic accident"
            elif "rain" in filename:
                auto_topic = "heavy rain"
            elif "heatwave" in filename:
                auto_topic = "heatwave"
            elif "landslide" in filename:
                auto_topic = "landslide"
            elif "storm" in filename:
                auto_topic = "storm"
            elif "pm10" in filename:
                auto_topic = "pm10"
            elif "water" in filename:
                auto_topic = "water accident"
            
            progress_html += f"<p>🏷️ Auto-detected: Category={auto_category}, Topic={auto_topic}</p>"
            yield processed_data, progress_html
            
            # 파일 로드 및 처리
            try:
                news_data = load_news_data(str(file_path))
                
                # 뉴스 데이터 전처리 (기존 로직 재사용)
                if news_data and isinstance(news_data, list) and len(news_data) > 0:
                    if "search_result" in list(news_data[0].keys()):
                        news_date_temp = [news["search_result"] for news in news_data]
                        news_data = [item for sublist in news_date_temp for item in sublist]
                    elif "item" in list(news_data[0].keys()):
                        news_data_before = news_data[0]["item"]["documentList"]
                        new_news_data = []
                        for d in news_data_before:
                            news_object = {
                                "date": d["date"],
                                "title": d["title"],
                                "text": d["content"],
                                "doc_id": d['docID'],
                                "url": d['url'],
                                "source": d['writerName'],
                                "vks": d['vks']
                            }
                            new_news_data.append(news_object)
                        news_data = new_news_data.copy()
                
                # 문서 처리 및 통합 인덱스 생성
                file_result = process_single_news_file_unified(
                    news_data, file_path.name, use_metadata, 
                    category_filter or auto_category, 
                    topic_filter or auto_topic
                )
                
                # 결과 처리
                if file_result.get("processed", False):
                    created_indexes = file_result.get("created_indexes", [])
                    processed_data["Number_of_indexes"] += len(created_indexes)
                    
                    # 인덱스 정보 업데이트 (통합 인덱스 지원)
                    for index_name in created_indexes:
                        if index_name.startswith("unified_"):
                            # 통합 인덱스 정보 저장
                            if "Index_by_unified" not in processed_data:
                                processed_data["Index_by_unified"] = {}
                            processed_data["Index_by_unified"][index_name] = file_result.get("documents", 0)
                            
                            # 메타데이터 통계도 저장
                            metadata_stats = file_result.get("metadata_stats", {})
                            processed_data["metadata_coverage"] = metadata_stats
                            
                        elif index_name.startswith("date_"):
                            date_key = index_name.replace("date_", "")
                            processed_data["Index_by_date"][date_key] = index_name
                        elif index_name.startswith("category_"):
                            cat_key = index_name.replace("category_", "")
                            processed_data["Index_by_category"][cat_key] = index_name
                        elif index_name.startswith("topic_"):
                            topic_key = index_name.replace("topic_", "")
                            processed_data["Index_by_topic"][topic_key] = index_name
                    
                    processed_data["Processed_files"].append({
                        "filename": file_path.name,
                        "size_mb": file_size_mb,
                        "documents": file_result.get("documents", 0),
                        "category": auto_category,
                        "topic": auto_topic,
                        "created_indexes": created_indexes,
                        "status": "success"
                    })
                    
                    total_docs += file_result.get("documents", 0)
                    progress_html += f"<p>✅ Created {len(created_indexes)} indexes from {file_path.name} ({file_result.get('documents', 0)} documents)</p>"
                    progress_html += f"<p>📂 Indexes: {', '.join(created_indexes)}</p>"
                else:
                    processed_data["Processed_files"].append({
                        "filename": file_path.name,
                        "size_mb": file_size_mb,
                        "documents": 0,
                        "category": auto_category,
                        "topic": auto_topic,
                        "error": file_result.get("error", "Unknown error"),
                        "status": "failed"
                    })
                    progress_html += f"<p style='color: red;'>❌ Failed to process {file_path.name}: {file_result.get('error', 'Unknown error')}</p>"
                
                yield processed_data, progress_html
                
            except Exception as e:
                error_msg = f"❌ Error processing {file_path.name}: {str(e)}"
                progress_html += f"<p style='color: red;'>{error_msg}</p>"
                yield processed_data, progress_html
                continue
        
        # 최종 결과
        time_spend = time.time() - start_time
        total_indexes = processed_data["Number_of_indexes"]
        
        status_msg = f"🎉 Auto-conversion completed (Efficient Method)!\n"
        status_msg += f"📊 Processed {len(files_to_process)} files, {total_docs} total documents\n"
        status_msg += f"🗂️ Created {total_indexes} FAISS indexes\n"
        
        # 통합 인덱스 정보 표시
        unified_indexes = processed_data.get("Index_by_unified", {})
        if unified_indexes:
            status_msg += f"🚀 Unified indexes: {len(unified_indexes)} (efficient storage)\n"
            for index_name, doc_count in unified_indexes.items():
                status_msg += f"   • {index_name}: {doc_count} documents\n"
            
            # 메타데이터 커버리지 표시
            metadata_coverage = processed_data.get("metadata_coverage", {})
            if metadata_coverage:
                status_msg += f"📊 Metadata coverage: {metadata_coverage.get('categories', 0)} categories, "
                status_msg += f"{metadata_coverage.get('topics', 0)} topics, {metadata_coverage.get('date_groups', 0)} date groups\n"
        else:
            # 기존 방식 정보
            status_msg += f"📅 Date indexes: {len(processed_data['Index_by_date'])}\n"
            status_msg += f"🏷️ Category indexes: {len(processed_data['Index_by_category'])}\n"
            status_msg += f"🔖 Topic indexes: {len(processed_data['Index_by_topic'])}\n"
        status_msg += f"⏱️ Time spent: {time_spend:.2f} seconds\n"
        status_msg += f"📁 Indexes saved to: {global_manager.base_dir}"
        
        # 현재 인덱스 상태 업데이트
        if global_manager:
            processed_data["Current_indices"] = global_manager.load_created_indexes()
        
        progress_html += f"<p style='color: green; font-weight: bold;'>{status_msg}</p>"
        
        # 생성된 인덱스 목록 표시
        if total_indexes > 0:
            progress_html += "<h4>📂 Created Indexes:</h4><ul>"
            for date_key, index_name in processed_data["Index_by_date"].items():
                progress_html += f"<li>📅 {index_name}</li>"
            for cat_key, index_name in processed_data["Index_by_category"].items():
                progress_html += f"<li>🏷️ {index_name}</li>"
            for topic_key, index_name in processed_data["Index_by_topic"].items():
                progress_html += f"<li>🔖 {index_name}</li>"
            progress_html += "</ul>"
        
        yield processed_data, progress_html
        
    except Exception as e:
        error_msg = f"❌ Auto-conversion failed: {str(e)}"
        progress_html += f"<p style='color: red;'>{error_msg}</p>"
        yield None, progress_html

def process_single_news_file_unified(news_data, filename, use_metadata, category_filter, topic_filter):
    """단일 뉴스 파일을 통합 인덱스로 처리"""
    global global_manager, global_api_key
    
    if not news_data:
        return {"processed": False, "documents": 0, "error": "No data"}
    
    try:
        # Document 객체 생성
        documents = []
        for idx, news in enumerate(news_data):
            # 메타데이터 추출
            if use_metadata and global_api_key:
                try:
                    category, topic = extract_category_topic(
                        news.get('text', ''), 
                        news.get('title', ''), 
                        global_api_key
                    )
                except:
                    category = category_filter or "unknown"
                    topic = topic_filter or "unknown"
            else:
                category = category_filter or "unknown"
                topic = topic_filter or "unknown"
            
            # Document 생성
            from langchain_core.documents import Document
            doc = Document(
                page_content=news.get('text', ''),
                metadata={
                    'title': news.get('title', ''),
                    'date': news.get('date', ''),
                    'source': news.get('source', ''),
                    'url': news.get('url', ''),
                    'category': category,
                    'topic': topic,
                    'doc_id': news.get('doc_id', f"{filename}_{idx}")
                }
            )
            documents.append(doc)
        
        if not documents:
            return {"processed": False, "documents": 0, "error": "No documents created"}
        
        # 통합 인덱스 생성 (모든 문서를 하나의 인덱스에)
        file_basename = filename.split('.')[0]
        unified_index_name = f"unified_{file_basename}"
        
        try:
            global_manager.create_index(unified_index_name, documents)
            created_indexes = [unified_index_name]
            print(f"✅ Created unified index: {unified_index_name} with {len(documents)} documents")
            
            # 메타데이터 통계 수집
            categories = set(doc.metadata.get('category', 'unknown') for doc in documents)
            topics = set(doc.metadata.get('topic', 'unknown') for doc in documents if doc.metadata.get('topic') != 'unknown')
            dates = set(doc.metadata.get('date', '')[:6] for doc in documents if len(doc.metadata.get('date', '')) >= 6)
            
            print(f"📊 Metadata coverage: {len(categories)} categories, {len(topics)} topics, {len(dates)} date groups")
            
            return {
                "processed": True, 
                "documents": len(documents),
                "created_indexes": created_indexes,
                "metadata_stats": {
                    "categories": len(categories),
                    "topics": len(topics),
                    "date_groups": len(dates)
                }
            }
        except Exception as e:
            print(f"❌ Error creating unified index {unified_index_name}: {e}")
            return {"processed": False, "documents": len(documents), "error": str(e)}
        
    except Exception as e:
        print(f"❌ Error in process_single_news_file_unified: {e}")
        return {"processed": False, "documents": 0, "error": str(e)}

def process_single_news_file(news_data, filename, use_metadata, category_filter, topic_filter):
    """단일 뉴스 파일을 처리하여 FAISS 인덱스 생성"""
    global global_manager, global_api_key
    
    if not news_data:
        return {"processed": False, "documents": 0, "error": "No data"}
    
    try:
        # Document 객체 생성
        documents = []
        for idx, news in enumerate(news_data):
            # 메타데이터 추출
            if use_metadata and global_api_key:
                try:
                    category, topic = extract_category_topic(
                        news.get('text', ''), 
                        news.get('title', ''), 
                        global_api_key
                    )
                except:
                    category = category_filter or "unknown"
                    topic = topic_filter or "unknown"
            else:
                category = category_filter or "unknown"
                topic = topic_filter or "unknown"
            
            # Document 생성
            from langchain_core.documents import Document
            doc = Document(
                page_content=news.get('text', ''),
                metadata={
                    'title': news.get('title', ''),
                    'date': news.get('date', ''),
                    'source': news.get('source', ''),
                    'url': news.get('url', ''),
                    'category': category,
                    'topic': topic,
                    'doc_id': news.get('doc_id', f"{filename}_{idx}")
                }
            )
            documents.append(doc)
        
        if not documents:
            return {"processed": False, "documents": 0, "error": "No documents created"}
        
        # 날짜별 인덱스 생성
        date_groups = {}
        for doc in documents:
            doc_date = doc.metadata.get('date', '')
            if len(doc_date) >= 6:  # YYYYMM 형식으로 그룹화
                date_key = doc_date[:6]  # YYYYMM
            else:
                date_key = "unknown"
            
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(doc)
        
        # 각 날짜 그룹별로 인덱스 생성
        created_indexes = []
        for date_key, date_docs in date_groups.items():
            index_name = f"date_{date_key}"
            try:
                global_manager.create_index(index_name, date_docs)
                created_indexes.append(index_name)
                print(f"✅ Created date index: {index_name} with {len(date_docs)} documents")
            except Exception as e:
                print(f"❌ Error creating date index {index_name}: {e}")
        
        # 카테고리별 인덱스 생성
        category_groups = {}
        for doc in documents:
            cat = doc.metadata.get('category', 'unknown')
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(doc)
        
        for cat, cat_docs in category_groups.items():
            index_name = f"category_{cat}"
            try:
                global_manager.create_index(index_name, cat_docs)
                created_indexes.append(index_name)
                print(f"✅ Created category index: {index_name} with {len(cat_docs)} documents")
            except Exception as e:
                print(f"❌ Error creating category index {index_name}: {e}")
        
        # 토픽별 인덱스 생성
        topic_groups = {}
        for doc in documents:
            topic = doc.metadata.get('topic', 'unknown')
            if topic and topic != 'unknown':
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(doc)
        
        for topic, topic_docs in topic_groups.items():
            index_name = f"topic_{topic.replace(' ', '_')}"
            try:
                global_manager.create_index(index_name, topic_docs)
                created_indexes.append(index_name)
                print(f"✅ Created topic index: {index_name} with {len(topic_docs)} documents")
            except Exception as e:
                print(f"❌ Error creating topic index {index_name}: {e}")
        
        return {
            "processed": True, 
            "documents": len(documents),
            "created_indexes": created_indexes,
            "date_groups": len(date_groups),
            "category_groups": len(category_groups),
            "topic_groups": len(topic_groups)
        }
        
    except Exception as e:
        print(f"❌ Error in process_single_news_file: {e}")
        return {"processed": False, "documents": 0, "error": str(e)}

def extract_category_topic(content, title, api_key):
    """GPT를 사용하여 콘텐츠에서 카테고리와 토픽 추출"""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        text_sample = f"Title: {title}\nContent: {content[:500]}..."  # 처음 500자만 사용
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a news categorization expert. Analyze the given news content and extract category and topic in JSON format.

                Instructions:
                - category: Choose ONE from ['politics', 'economics', 'society', 'culture', 'technology', 'sports', 'entertainment', 'disaster', 'crime', 'health', 'environment', 'international', 'other']
                - topic: A specific keyword or phrase (max 2 words) that best describes the main subject
                
                Rules:
                - Always respond with valid JSON only
                - Be precise and concise
                - Choose the most relevant category
                
                Example responses:
                {"category": "disaster", "topic": "fire"}
                {"category": "disaster", "topic": "earthquake"}
                {"category": "disaster", "topic": "heavy snow"}
                {"category": "disaster", "topic": "infection"}
                {"category": "disaster", "topic": "traffic accident"}"""},
                {"role": "user", "content": f"Categorize this news content:\n\n{text_sample}"}
            ],
            max_tokens=80,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result.get("category", "other"), result.get("topic", "general")
        
    except Exception as e:
        print(f"Category/Topic extraction error: {e}")
        return "other", "general"

# FAISS DB에 넣기
def process_uploaded_files(files, use_type='Retriever with Metadata', question_type='Generate', use_faiss=True):
    global global_manager, global_api_key
    if global_manager is None:
        global_manager = initialize_manager()

    if not files:
        return None, "Please upload at least one file."

    processed_data = {
        "Number_of_indexes": len(files),
        "Index_by_date": {},
        "Current_indices": {}
    }

    start_time = time.time()

    total_docs = 0
    progress_html = ""
    processed_list = [] # tlfgod

    # 함수 호출
    from utils import process_openai_generate, process_openai_mcq

    try:
        for file_idx, file in enumerate(files, 1):

            search_interface = SearchInterface(global_manager)
            search_interface.openai_api_key = global_api_key

            import openai
            client = openai.OpenAI(api_key=global_api_key)

            progress_html += f"<p>Processing File {file_idx}/{len(files)}: {file.name}, Process : {use_type}</p>"
            yield processed_data, progress_html

            news_data = load_news_data(file.name)

            # news_data key analysis
            assert isinstance(news_data, list), "News data should be a list."
            assert all(isinstance(news, dict) for news in news_data), "Each news item should be a dictionary."
            print(news_data[0].keys())

            # news_data를 부분만 추출
            if "search_result" in list(news_data[0].keys()):
                    print("NEWSKEYS")
                    assert "text" in news_data[0]["search_result"][0], "뉴스 데이터의 'search_result' 항목에 'text' 키가 있어야 합니다."
                    news_date_temp = [news["search_result"] for news in news_data]
                    news_data = [item for sublist in news_date_temp for item in sublist]  # flatten list
                    with open("news_data_temp_flattened.json", "w", encoding="utf-8") as f:
                        json.dump(news_data, f, ensure_ascii=False, indent=4)
            elif "item" in list(news_data[0].keys()):
                # form {success, code, message item -> {keyword, totalCnt, documentList}}
                assert "content" in news_data[0]["item"]["documentList"][0], "뉴스 데이터의 item 항목에 'content' 키가 있어야 합니다."
                news_data_before = news_data[0]["item"]["documentList"] # 일단 정리
                new_news_data = []
                # 풀기 쉽게 재정리
                for d in news_data_before:
                    news_object = {
                        "date": d["date"],
                        "title": d["title"],
                        "text": d["content"],
                        "doc_id": d['docID'],
                        "url": d['url'],
                        "source": d['writerName'],
                        "vks": d['vks']
                    }
                    new_news_data.append(news_object)
                news_data = new_news_data.copy()

            # 분석 시계열 바꾸기
            analyzed_news_data = []
            for news_idx, news in enumerate(tqdm(news_data, desc="Processing news data")):
                analyzed_news = analyze_qa_type(news, qa_name="realtimeqa", question_type=question_type, use_type=use_type)
                analyzed_news_data.append(analyzed_news)

            use_metadata = True if use_type.lower() in ['retriever with metadata', 'no retriever with metadata'] else False
            use_retriever = True if use_type.lower() in ['retriever with metadata', 'retriever with no metadata', 'retriever only'] else False

            # use_type에 따라 처리 방식 결정 - Retrieve로 시작할 때 저장
            # use_retriever가 True이면 검색 결과 저장
            search_list = [] # 각 뉴스마다 대응하기 analyzed news_data와 1:1 대응
            if use_retriever:
                search_interface.retriever = global_retriever
                search_interface.retriever_model = global_retriever_model
                search_interface.tokenizer = global_tokenizer

                for news_idx, news in enumerate(tqdm(analyzed_news_data, desc="Processing news data")):
                    print(f"[DEBUG] Analyzed news for index {news_idx}: {news}")
                    # query -> gcs_search 사용해서
                    query = news.get("query", "")
                    if not query:
                        search_list.append([])
                        continue
                    # find search_result from query by retrieve_single_question
                    top_k = 5
                    # use_metadata에 따라 날짜 설정
                    if use_metadata:
                        end_date = news.get('date', '20000101')
                        if isinstance(end_date, str):
                            end_date = process_date_string(end_date)
                        elif isinstance(end_date, datetime.date):
                            end_date = end_date.strftime("%Y%m%d")
                        else:
                            end_date = '20000101_nm'
                        if re.match(r"20[0-2][0-9][01][0-9][0-3][0-9]", end_date):
                            start_date = compute_relative_date(end_date, -30)  # 30일 전
                        else:
                            start_date = None
                        # query에서도 날짜 정보 사용해서
                        # query_context = return_date_info(query, use_heidel_time=False, use_llm=True)
                        # query = f"Time Metadata : {query_context} is given. \nNow answer the question with given metadata  {query}" # 질문에 time_metadata 정보 삽입
                    else:
                        end_date = '20000101_nm'
                        start_date = None

                    # BM25 강화 검색 사용
                    try:
                        from utils import retrieve_single_question_with_bm25
                        search_result = retrieve_single_question_with_bm25(
                            query, global_retriever_model, global_retriever, global_tokenizer, GCS_KEY, ENGINE_KEY,
                            top_k=10, start_date=start_date, end_date=end_date, use_metadata=use_metadata,
                            use_reranking=True, use_bm25=True,
                            rerank_method="cohere" if COHERE_API_KEY else "custom",
                            rerank_api_key=COHERE_API_KEY or OPENAI_API_KEY,
                            chunk_size=1000, chunk_overlap=500
                        )
                        print(f"🚀 BM25-enhanced search completed for query: {query}")
                    except ImportError:
                        # BM25 라이브러리가 없으면 기본 방식 사용
                        print("⚠️ BM25 not available, using standard retrieval")
                        search_result = retrieve_single_question(
                            query, global_retriever_model, global_retriever, global_tokenizer, GCS_KEY, ENGINE_KEY,
                            top_k=10, start_date=start_date, end_date=end_date, use_metadata=use_metadata,
                            use_reranking=True,
                            rerank_method="cohere" if COHERE_API_KEY else "custom",
                            rerank_api_key=COHERE_API_KEY or OPENAI_API_KEY,
                            chunk_size=1000, chunk_overlap=500
                        )

                    if not search_result:
                        print(f"No search result for query: {query}")
                        search_list.append([])
                        continue
                    else:
                        print(f"Search result for query '{query}': {search_result}")
                        search_list.append(search_result)

                # search_list -> FAISS DB에 저장. 중복이 있을 수 있으므로 불러오기
                if use_metadata and use_faiss:
                    # 날짜별로 인덱싱 및 문서 처리
                    progress_html_new = progress_html + f"<p>Creating Indexes... ({len(search_list)})</p>"
                    yield processed_data, progress_html_new

                    # search_list -> 문서 생성
                    documents = []
                    for idx, (search_result, news) in enumerate(tqdm(zip(search_list, analyzed_news_data), desc="Creating documents from search results")):
                        if not search_result:
                            logging.warning(f"No search results for index {idx} in {file.name}. Skipping.")
                            continue
                        for result in search_result:
                            doc = Document(
                                page_content=result.get('text', 'No Text'),
                                metadata={
                                    'title': result.get('title', 'No Title'),
                                    'doc_id': result.get('doc_id', 'No Doc ID'),
                                    'query': query,
                                    'date': result.get('date', '20000101'),
                                    'source': result.get('source', 'None')
                                }
                            )
                            documents.append(doc)
                            # 문서명 호출 - faiss_indexes_metadata/{date}_{source} - example : 20250601_CNN
                            logging.info(f"Document created for query '{query}': {doc.metadata['title']} (ID: {doc.metadata['doc_id']})")
                            index_name = f"{result.get('date', '20000101')}_{result.get('source', 'None').replace(' ', '_')}"
                            
                            # 인덱스가 없으면 생성, 있으면 문서 추가
                            if not global_manager.has_index(index_name):
                                global_manager.create_index(index_name, documents=[doc])
                                print(f"✅ Created new index: {index_name}")
                            else:
                                # 기존 인덱스에 문서 추가 (이 부분은 별도 구현 필요)
                                logging.info(f"Index {index_name} already exists. Adding documents to it.")

                            processed_data["Current_indices"][index_name] = len(documents)
                            total_docs += len(documents)

                            progress_html_new = progress_html + f"<p>Added {len(documents)} documents to the index.</p>"
                            yield processed_data, progress_html_new
                            print(f"Total documents in index {index_name}: {len(documents)}")

                # 메타데이터 없을때는 20000000_nm 인덱스에 추가
                elif not use_metadata and use_faiss:
                    index_name = "20000000_nm"
                    # documents 재정의
                    documents = []
                    # if not global_manager.has_index(index_name):
                    #    global_manager.create_index(index_name, documents=[], ids=[])
                    # documents 생성 -
                    for idx, (search_result, news) in enumerate(tqdm(zip(search_list, analyzed_news_data), desc="Creating documents from search results")):
                        if not search_result:
                            logging.warning(f"No search results for index {idx} in {file.name}. Skipping.")
                            continue
                        for result in search_result:
                            doc = Document(
                                page_content=result.get('text', 'No Text'),
                                metadata={
                                    'title': result.get('title', 'No Title'),
                                    'doc_id': result.get('doc_id', 'No Doc ID'),
                                    'query': query,
                                    'date': '20000000_nm',
                                    'source': 'None'
                                }
                            )
                            documents.append(doc)
                            # 문서명 호출 - faiss_indexes_metadata/{date}_{source} - example : 20250601_CNN
                            logging.info(f"Document created for query '{query}': {doc.metadata['title']} (ID: {doc.metadata['doc_id']})")
                    global_manager.create_index(index_name, documents)
                    processed_data["Current_indices"][index_name] = len(documents)
                    total_docs += len(documents)
                    progress_html_new = progress_html + f"<p>Added {len(documents)} documents to the index.</p>"
                    yield processed_data, progress_html_new
                    print(f"✅ Created index {index_name} with {len(documents)} documents")


            # 답변 처리 - 모든 경우에 해결
            # elif use_type.lower() in ['qa', 'realtimeqa', 'realtime', 'cnnqa', 'newsqa']:
            answers = [] # 답변 목록
            scores = []
            answer_objs = [] # 전체 목록
            context = "" # 우선 가져오지 않을 때는 빈문자열로 처리


            progress_html += f"<p>Starting QA processing for {len(news_data)} questions...</p>"
            yield processed_data, progress_html


            for news_idx, news in enumerate(tqdm(analyzed_news_data, desc="Processing news data")):
                res_obj = dict() # 답변 형식 확인하기
                # key - id, query, answers
                # 답변 구하기
                query = news.get('query') #
                if not query:
                    print(f"No query found for news: {news}")
                    continue

                # context 정의
                if use_retriever:
                    # 검색 결과가 있는 경우
                    search_result = search_list[news_idx]
                    if not search_result:
                        print(f"No search result for query: {query}")
                        continue
                    context = "\n".join([f"{item.get('title', 'idea')}: {item.get('text', '')}" for item in search_result])
                else:
                    # 검색 결과가 없는 경우 빈 문자열로 설정
                    if use_metadata:
                        # 메타데이터가 있으면 context를 chatgpt의 메타데이터 입력 함수 사용.
                        context = return_date_info(query, use_heidel_time=True, use_llm=True)
                    else:
                        context = ""

                res_obj['id'] = news.get('id', f"{news_idx}_{uuid4().hex[:8]}")  # id 생성
                res_obj['query'] = query
                res_obj['score'] = 0.5 # 기본 점수 설정
                # find answer from query
                if question_type.lower() == "generate":
                    # 생성형 질문
                    answer = process_openai_generate(
                        query,
                        context,
                        client=client
                    )
                elif question_type.lower() == "mcq":
                    # 선다형 질문
                    answer = process_openai_mcq(
                        query,
                        context,
                        choices=news.get('choices', []),
                        client=client
                    )
                else:
                    print(f"Unsupported question type: {question_type}")
                    continue

                if not answer:
                    answer = ["0"]  # 기본값 설정
                elif isinstance(answer, str):
                    answer = [answer]

                res_obj['answer'] = answer
                res_obj['prediction'] = answer  # 예측 결과로 답변 사용
                answers.append(answer)
                answer_objs.append(res_obj)

                part_progress_html = progress_html + f"<p>Processed question {news_idx + 1}/{len(analyzed_news_data)}: {query}</p>"
                yield processed_data, part_progress_html



            # 정확도 accuracy 사용
            try:
                print(f"Evaluating {len(answer_objs)} answers against {len(news_data)} news data...")
                print(answer_objs[:5])  # 디버깅용 출력
                print(news_data[:5])  # 디버깅용 출력

                #pred length
                print("Length of answer_objs and news_data:", len(answer_objs), len(news_data))

                # global_generator_model이 리스트인 경우 첫 번째 요소 사용
                model_name = global_generator_model
                if isinstance(global_generator_model, list):
                    model_name = global_generator_model[0] if global_generator_model else "openai mcq"
                elif not isinstance(global_generator_model, str):
                    model_name = str(global_generator_model)

                if question_type == "MCQ":
                    eval_results = accuracy(answer_objs, news_data) # 정확도 선다형
                elif question_type == "Generate":
                    eval_results = gen_eval(answer_objs, news_data) # 주관식

                # 결과를 문자열로 변환하여 저장
                if isinstance(eval_results, dict):
                    # 딕셔너리인 경우 주요 정보만 추출
                    total_questions = eval_results.get('total', len(news_data))
                    if question_type == "MCQ":
                        accuracy_score = eval_results.get('accuracy', 'N/A')
                        correct_answers = eval(accuracy_score) * int(total_questions) if isinstance(accuracy_score, str) else accuracy_score * int(total_questions)
                        # correct_answers = eval_results.get('correct', 'N/A')
                        result_str = f"File: {file.name} - Accuracy: {accuracy_score}, Correct: {correct_answers}/{total_questions}"
                    elif question_type == "Generate":
                        em_score = eval_results.get('em', 0)
                        f1_score = eval_results.get('f1', 0)
                        result_str = f"File: {file.name} - EM: {em_score}, F1: {f1_score}, Total: {total_questions}"

                else:
                    result_str = f"File: {file.name} - Result: {str(eval_results)}"

                processed_list.append(result_str)
                print(f"✅ Evaluation completed for {file.name}: {result_str}")

                if question_type == "MCQ":
                    accuracy_report_file = f"results/accuracy_report_{file.name.split('/')[-1]}"
                elif question_type == "Generate":
                    accuracy_report_file = f"results/accuracy_report_{file.name.split('/')[-1].replace('.json', '_gen.json')}"
                accuracy_reports = make_accuracy_reports(answer_objs, news_data, file_name=accuracy_report_file)
                print("accuracy_reports")
                for report in accuracy_reports[:5]:
                    print(report)

            except Exception as eval_error:
                error_str = f"File: {file.name} - Evaluation Error: {str(eval_error)}"
                processed_list.append(error_str)
                print(f"⚠️ Evaluation failed for {file.name}: {eval_error}")


        status_msg = f"? Processed Finished:\n {len(files)} files, {total_docs} documents processed."

        status_msg += f"\n{len(processed_list)} evaluations completed."
        time_spend = time.time() - start_time # 소요시간 (초로 표현)
        if processed_list:
            joined_list = '\n'.join(processed_list)
            status_msg += f"\nEvaluation Results:\n{joined_list} \nTime spent: {time_spend:.2f} seconds"

        progress_html += f"<p style='color: green;'>{status_msg}</p>"

        # 모든 경우에 yield로 반환 (일관성 유지)
        yield processed_data, progress_html


    except Exception as e:
        error_msg = f"? 처리 중 오류 발생: {str(e)}"
        progress_html += f"<p style='color: red;'>{error_msg}</p>"
        yield None, progress_html


def make_accuracy_reports(pred_data, gold_data, file_name="results/metadata_extraction.jsonl"):
    """pred_data의 정답 결과와 gold_data의 정답 결과를 비교하여 정확도 보고서를 생성하는 함수"""
    assert len(pred_data) == len(gold_data), "Prediction and gold data must have the same length."
    accuracy_results = []
    obj_format = {
        "question_id": "",
        "type": "mcq", #mcq or generate
        "prediction": [],
        "answer": [],
        "em": 0, # exact match or choice match(correct choice)
        "f1": 0, # f1 score
        "score": 0.0, # score
    }
    for pred, gold in zip(pred_data, gold_data):
        from evaluate import exact_match_score, f1_score
        import itertools
        res_obj = obj_format.copy()
        res_obj["question_id"] = pred.get("question_id", gold.get("question_id", "unknown"))
        res_obj["prediction"] = pred.get("prediction", [])
        res_obj["type"] = "mcq" if str(res_obj["prediction"][0]).isnumeric() else "generate" # 타입
        res_obj["score"] = float(pred.get("score", 0.0))  # score 값 설정
        if res_obj["type"] == "mcq":
            res_obj["answer"] = gold.get("answer", [])
            res_obj["em"] = int(res_obj["prediction"][0] in res_obj["answer"])
        else:
            answer_choices = gold.get("choices", [])
            answer_num = gold.get("answer", [])
            if not answer_choices or not answer_num:
                res_obj["answer"] = []
            else:
                pred = pred.get("prediction", [""])
                golds = [gold["choices"][int(idx)] for idx in gold["answer"]]
                golds = [' '.join(perm) for perm in list(itertools.permutations(golds))]
                res_obj["answer"] = [answer_choices[int(num)] for num in answer_num]
                res_obj["em"] = exact_match_score(pred, golds)
                res_obj["f1"] = f1_score(pred, golds)
        accuracy_results.append(res_obj)

    accuracy_objs = [json.dumps(res, ensure_ascii=False) for res in accuracy_results]

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("\n".join(accuracy_objs))

    print(f"Accuracy report saved to {file_name}")
    return accuracy_results

# 인터페이스 생성
def create_gradio_interface():
    global global_manager, global_api_key

    # OpenAI API 키 초기화
    global_api_key = OPENAI_API_KEY

    global_manager = initialize_manager()
    search_interface = SearchInterface(global_manager)
    # 최대 길이 지정
    max_length = 4000  # 최대 길이 설정 (토큰 수에 따라

    # Gradio 인터페이스 설정
    with gr.Blocks(title="Find Data from Query") as demo:
        gr.Markdown("""## Find Data from Query""")

        # 시스템 상태 표시
        with gr.Row():
            system_status = gr.HTML(value=f"""
                <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                    <b>System Status:</b><br>
                    💻 Device: {'🟢 GPU (CUDA)' if torch.cuda.is_available() else '🟡 CPU'}<br>
                    🧠 Models: {'🟢 Loaded' if global_retriever else '🔴 Not Loaded'}<br>
                    📚 Embeddings: {'🟢 Ready' if global_manager else '🔴 Not Ready'}
                </div>
            """)

        with gr.Group(visible=True) as api_settings:
            with gr.Row():
                api_key = gr.Textbox(label="OpenAI API Key", type="password")
                google_api_key = gr.Textbox(label="Google API Key (Optional)", type="password", placeholder="For hybrid search")
            with gr.Row():
                google_cse_id = gr.Textbox(label="Google CSE ID (Optional)", placeholder="For hybrid search")

        model_status = gr.Textbox(label="Model status", interactive=False)
        init_model_btn = gr.Button("Initialize Model")

        # 실험 설정 섹션 (QA 평가용)
        gr.Markdown("### Experiment Configuration")
        with gr.Row():
            with gr.Column():
                exp_type = gr.Radio(
                    choices=["No Retriever and No Metadata", "No Retriever with Metadata", "Retriever Only", "Retriever with Metadata"],
                    label="Select the type of experiment",
                    value="Retriever with Metadata"
                )
                question_type = gr.Radio(
                    choices=["MCQ", "Generate"],
                    label="Select the type of question",
                    value="Generate"
                )
                # 날짜 범위 설정
                date_range = gr.Textbox(label="Date Range - YYMMDD/YYMMDD")

            with gr.Column():
                file_output = gr.JSON(label="Current File Status")
                upload_button = gr.File(
                    label="Upload JSON/JSONL files for QA Evaluation",
                    file_types=[".json", ".jsonl"],
                    file_count="multiple",
                )
                status_output = gr.HTML(label="Status Output")

        # FAISS 변환 섹션
        gr.Markdown("### FAISS Index Conversion")
        with gr.Row():
            gr.Markdown("Convert news data to FAISS indexes with metadata-based organization")
            
        with gr.Row():
            with gr.Column():
                # 자동 변환 옵션
                auto_convert_news_data = gr.Checkbox(
                    label="Auto-convert news_data/ folder",
                    value=True
                )
                faiss_use_metadata = gr.Checkbox(
                    label="Use Metadata Extraction",
                    value=True
                )
                faiss_category_filter = gr.Dropdown(
                    label="Filter by Category",
                    choices=["all", "disaster", "crime", "politics", "economics", "society", "culture", "technology", "sports", "entertainment", "health", "environment", "international", "other"],
                    value="disaster"
                )
                faiss_topic_filter = gr.Textbox(
                    label="Filter by Topic",
                    placeholder="e.g., fire, earthquake, infection (leave empty for all)"
                )
                
            with gr.Column():
                # 뉴스 데이터 파일 선택
                news_data_files = gr.CheckboxGroup(
                    label="Select News Data Files",
                    choices=[],
                    value=[]
                )
                refresh_news_files_btn = gr.Button("🔄 Refresh News Files", variant="secondary")
                
                # 수동 업로드 (옵션)
                faiss_upload_button = gr.File(
                    label="Manual Upload (Optional)",
                    file_types=[".json", ".jsonl"],
                    file_count="multiple"
                )
                
                convert_to_faiss_btn = gr.Button("🚀 Convert to FAISS Indexes", variant="primary")
                faiss_output = gr.JSON(label="FAISS Conversion Status")
                faiss_status_output = gr.HTML(label="Conversion Progress")

        # 검색 섹션
        with gr.Row():
            with gr.Column():
                # 쿼리 검색
                query_input = gr.Textbox(
                    label="Please enter your query",
                    placeholder="Example : What is the cause of the fire in the mixed-use building on December 31, 2023?",
                    elem_classes=["submit-on-enter"],
                    autofocus=True
                )
                
                # 예시 질문 (질문 바로 아래 배치)
                gr.Examples(
                    examples=[
                        ["지난 3년간 대형 화재 사건들 알려줘"],
                        ["폭설 재해와 그 영향에 대해 알려줘."],
                        ["최근 지진 소식과 피해 보고를 알려줘."],
                        ["2021~2022년 코로나19 감염 추세를 알려줘."],
                        ["교통사고 통계와 원인에 대해 알려줘."],
                        ["폭염 재해와 건강에 미치는 영향을 알려줘."],
                        ["집중호우 및 홍수 발생 사례를 알려줘."],
                        ["2021~2022년 재난 관련 범죄 사건을 알려줘."],
                        ["미세먼지(PM10) 대기 오염과 건강 영향에 대해 알려줘."],
                        ["산사태 재해와 예방 대책에 대해 알려줘."]
                    ],
                    inputs=[query_input],
                    label="예시 질문 (클릭하여 사용)"
                )

                with gr.Row():
                    # 날짜 범위 표시
                    date_range_display = gr.Textbox(
                        label="Detected Date Range",
                        interactive=False,
                        placeholder="Date range will be shown here"
                    )

                    # top_k 슬라이더
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="🔢 Number of Results (Top-K)",
                        info="Select how many search results to retrieve and display"
                    )

                # 필터링 옵션
                with gr.Row():
                    gr.Markdown("### Filtering Options")

                with gr.Row():
                    use_date_filter = gr.Checkbox(
                        label="Use Date Filtering",
                        value=True
                    )
                    use_category_filter = gr.Checkbox(
                        label="Use Category Filtering",
                        value=True
                    )
                    use_topic_filter = gr.Checkbox(
                        label="Use Topic Filtering",
                        value=True
                    )

                # 카테고리 및 토픽 선택
                with gr.Row():
                    category_select = gr.Dropdown(
                        label="Select Category",
                        choices=["all", "politics", "economics", "society", "culture", "technology", "sports", "entertainment", "disaster", "crime", "health", "environment", "international", "other"],
                        value="all"
                    )
                    topic_select = gr.Dropdown(
                        label="Select Topic", 
                        choices=["all"],
                        value="all"
                    )
                    
                # 토픽 목록 업데이트 버튼
                update_filters_btn = gr.Button("Update Available Filters", variant="secondary")

                # 하이브리드 검색 옵션
                with gr.Row():
                    use_hybrid_search = gr.Checkbox(
                        label="Use Hybrid Search (FAISS + Google)",
                        value=True
                    )
                    generate_comprehensive_answer = gr.Checkbox(
                        label="Generate Comprehensive Answer",
                        value=True
                    )

                # 검색 버튼
                with gr.Column():
                    search_button = gr.Button("Search", variant="primary")
                    hybrid_search_button = gr.Button("Hybrid Search + Answer", variant="primary")

                # 경고 메시지 출력
                warning_output = gr.HTML()


        # 검색 결과 설명 섹션
        gr.Markdown("### Explanation of Search Results")
        with gr.Row():
            results_output = gr.Textbox(
                label="News Search Results",
                lines=10,
                show_copy_button=True
            )

        # 인덱스 관리 섹션
        with gr.Row():
            with gr.Column():
                index_info_button = gr.Button("Check Index Status", variant="secondary")
                index_info_output = gr.JSON(label="Index Status")



        # 함수 정의들
        def update_index_info():
            return global_manager.load_created_indexes()
        # Enter key submission handler


        def init_model(api_key=None, google_key=None, google_cse=None):
            global global_api_key, global_hybrid_search
            try:
                result = search_interface.init_openai_model(api_key)
                global_api_key = api_key  # OpenAI API 키 설정
                
                # Google API 키 업데이트
                if global_hybrid_search and (google_key or google_cse):
                    global_hybrid_search.google_api_key = google_key
                    global_hybrid_search.google_cse_id = google_cse
                    result += f"\n🌐 Google Search API updated: {'✅' if google_key and google_cse else '⚠️ Incomplete'}"
                
                return result

            except Exception as e:
                return f"An error occurred during loading: {str(e)}"


        def hybrid_search_with_answer(query, use_date_filter, use_category_filter, use_topic_filter, 
                                     category_select, topic_select, use_hybrid, generate_answer, top_k):
            """하이브리드 검색 + 종합 답변 생성"""
            import time
            search_start_time = time.time()
            
            if not global_hybrid_search:
                return "Error: Hybrid search engine not initialized", "No date range available"
            
            # 메타데이터 추출
            results, date_range, metadata = return_date_info(query)
            
            try:
                # 필터링 옵션 준비
                search_category = None
                search_topic = None
                search_date_range = None

                if use_date_filter:
                    search_date_range = date_range

                if use_category_filter:
                    if category_select and category_select != "all":
                        search_category = category_select
                    elif metadata["category"]:
                        search_category = metadata["category"]

                if use_topic_filter:
                    if topic_select and topic_select != "all":
                        search_topic = topic_select
                    elif metadata["topic"]:
                        search_topic = metadata["topic"]

                # 하이브리드 검색 실행 - 모든 필터가 비활성화된 경우 기본 검색
                if not use_date_filter and not use_category_filter and not use_topic_filter:
                    # 모든 필터가 비활성화된 경우: 전체 검색
                    search_results = global_hybrid_search.search_with_fallback(
                        query=query,
                        k=int(top_k),
                        date_info=None,  # 전체 기간
                        category=None,
                        topic=None,
                        use_google=use_hybrid
                    )
                else:
                    # 필터가 활성화된 경우: 필터 적용 검색
                    search_results = global_hybrid_search.search_with_fallback(
                        query=query,
                        k=int(top_k),
                        date_info=search_date_range,
                        category=search_category,
                        topic=search_topic,
                        use_google=use_hybrid
                    )

                # 날짜 표시 설정
                if not use_date_filter and not use_category_filter and not use_topic_filter:
                    date_display = "Hybrid search: All available data (no filters applied)"
                    if use_hybrid:
                        date_display += " [Google Search enabled]"
                else:
                    # 날짜 표시 형식 변환
                    start_date, end_date = date_range.split('/')
                    formatted_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
                    formatted_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
                    
                    active_filters = []
                    if use_date_filter:
                        active_filters.append("Date")
                    if search_category:
                        active_filters.append(f"Category: {search_category}")
                    if search_topic:
                        active_filters.append(f"Topic: {search_topic}")
                    if use_hybrid:
                        active_filters.append("Google Search")

                    date_display = f"Hybrid search from {formatted_start} to {formatted_end}"
                    if active_filters:
                        date_display += f" [Filters: {', '.join(active_filters)}]"

                # 종합 답변 생성 또는 기본 결과 표시
                if generate_answer and global_api_key:
                    import openai
                    client = openai.OpenAI(api_key=global_api_key)
                    comprehensive_answer = create_comprehensive_answer(search_results, query, client)
                    results = f"### 🤖 AI 종합 답변\n\n{comprehensive_answer}\n\n"
                else:
                    results = "### 🔍 하이브리드 검색 결과\n\n"
                
                # 검색 결과 상세 표시 (점수 높은 순서대로)
                if not search_results:
                    results += "검색 결과가 없습니다. 다른 키워드로 시도해보세요.\n"
                else:
                    # 점수 기준으로 역순 정렬 (높은 점수부터)
                    search_results_sorted = sorted(search_results, key=lambda x: x.get('score', 0), reverse=True)
                    
                    results += "### 📋 상세 검색 결과 (점수 높은 순서)\n\n"
                    results += f"**총 {len(search_results_sorted)}개 결과 발견 (관련성 높은 순으로 정렬)**\n\n"
                    
                    for idx, result in enumerate(search_results_sorted, 1):
                        title = result.get('title', 'No Title')
                        content = result.get('content', '')[:200] + "..."
                        source = result.get('source', 'Unknown')
                        search_type = result.get('search_type', 'unknown')
                        date = result.get('date', '')
                        score = result.get('score', 0)
                        
                        # 하이브리드 점수 정보 (있는 경우)
                        hybrid_score = result.get('hybrid_score', score)
                        bm25_score = result.get('bm25_score', 0)
                        
                        # 날짜 형식 변환
                        if len(date) == 8:
                            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
                        
                        # 검색 타입별 아이콘
                        type_icon = "🗃️" if search_type.startswith("faiss") else "🌐" if search_type == "google" else "📄"
                        
                        # 순위 표시 (1위는 🥇, 2위는 🥈, 3위는 🥉)
                        rank_icon = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"#{idx}"
                        
                        results += f"{rank_icon} {type_icon} **{title}**\n"
                        
                        # 점수 정보 (하이브리드 점수가 있으면 상세 표시)
                        if bm25_score > 0:
                            results += f"🎯 **Hybrid Score: {hybrid_score:.4f}** (Vector: {score:.3f} + BM25: {bm25_score:.3f})\n"
                        else:
                            results += f"⭐ **Score: {score:.4f}**\n"
                        
                        # 출처 정보를 더 명확히 표시
                        results += f"📅 {date} | 📰 **출처: {source}** | 🔍 **검색유형: {search_type}**\n"
                        results += f"📝 {content}\n"
                        results += "─" * 60 + "\n\n"


                return results, date_display

            except Exception as e:
                return f"하이브리드 검색 중 오류 발생: {str(e)}", "Error in processing"

        def search_with_metadata_filters(query, use_date_filter, use_category_filter, use_topic_filter, category_select, topic_select, top_k):
            # 성능 모니터링 시작
            import time
            search_start_time = time.time()
            
            # 메타데이터 추출 (날짜, 카테고리, 토픽)
            results, date_range, metadata = return_date_info(query)

            if not global_manager:
                return "Error: Manager not initialized", "No date range available"

            # 검색 실행
            try:
                start_date, end_date = date_range.split('/')
                # 날짜 표시 형식 변환 (YYYYMMDD -> YYYY-MM-DD)
                formatted_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
                formatted_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

                # 필터링 옵션 준비
                active_filters = []
                search_category = None
                search_topic = None
                search_date_range = None

                if use_date_filter:
                    active_filters.append("Date")
                    search_date_range = date_range

                if use_category_filter:
                    if category_select and category_select != "all":
                        search_category = category_select
                        active_filters.append(f"Category: {search_category}")
                    elif metadata["category"]:
                        search_category = metadata["category"]
                        active_filters.append(f"Category: {search_category}")

                if use_topic_filter:
                    if topic_select and topic_select != "all":
                        search_topic = topic_select
                        active_filters.append(f"Topic: {search_topic}")
                    elif metadata["topic"]:
                        search_topic = metadata["topic"]
                        active_filters.append(f"Topic: {search_topic}")

                date_display = f"Searching from {formatted_start} to {formatted_end}"
                if active_filters:
                    date_display += f" [Filters: {', '.join(active_filters)}]"

                # 검색 실행 - 모든 필터가 비활성화된 경우 기본 검색 수행
                if not use_date_filter and not use_category_filter and not use_topic_filter:
                    # 모든 필터가 비활성화된 경우: 날짜 필터 없이 전체 검색
                    search_results = global_manager.search_without_date(
                        query=query,
                        k=int(top_k)
                    )
                    date_display = "Searching all available data (no filters applied)"
                else:
                    # 필터가 활성화된 경우: 메타데이터 필터 검색
                    search_results = global_manager.search_with_metadata_filters(
                        query=query,
                        k=int(top_k),
                        date_info=search_date_range,
                        category=search_category,
                        topic=search_topic,
                        use_metadata=use_date_filter
                    )

                # 검색 결과 포맷팅 (점수 높은 순서대로)
                results += "\n### 🔍 Search Results (점수 높은 순서)\n"
                if not search_results:
                    results += "No results found with the specified filters.\n"
                else:
                    # 점수 기준으로 역순 정렬 (높은 점수부터)
                    search_results_sorted = sorted(search_results, key=lambda x: x[1], reverse=True)
                    
                    results += f"**총 {len(search_results_sorted)}개 결과 발견 (관련성 높은 순으로 정렬)**\n\n"
                    
                    for idx, (doc, score) in enumerate(search_results_sorted, 1):
                        doc_date = doc.metadata.get('date', 'Unknown')
                        if len(doc_date) == 8:  # YYYYMMDD 형식이면 변환
                            doc_date = f"{doc_date[:4]}-{doc_date[4:6]}-{doc_date[6:]}"

                        # 순위 아이콘
                        rank_icon = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"#{idx}"
                        
                        results += f"{rank_icon} **{doc.metadata.get('title', 'No Title')}**\n"
                        results += f"🎯 **Score: {score:.4f}**\n"
                        # 출처 정보를 더 명확히 표시
                        doc_source = doc.metadata.get('source', 'Unknown')
                        results += f"📅 {doc_date} | 📰 **출처: {doc_source}** | 🏷️ {doc.metadata.get('category', 'N/A')} | 🔖 {doc.metadata.get('topic', 'N/A')}\n"
                        results += f"📝 {doc.page_content[:200]}...\n"
                        results += "─" * 60 + "\n\n"


            except Exception as e:
                results += f"\nError in search: {str(e)}"
                date_display = "Error in date processing"

            return results, date_display

        def update_available_filters():
            """사용 가능한 카테고리와 토픽 목록 업데이트"""
            if global_manager:
                categories = ["all"] + global_manager.get_available_categories()
                topics = ["all"] + global_manager.get_available_topics()
                return gr.update(choices=categories), gr.update(choices=topics)
            return gr.update(), gr.update()

        def refresh_news_files():
            """뉴스 데이터 파일 목록 새로고침"""
            try:
                files = get_news_data_files()
                choices = [file_info for _, file_info in files]
                values = [file_path for file_path, _ in files]
                return gr.update(choices=list(zip(values, choices)), value=[])
            except Exception as e:
                print(f"Error refreshing news files: {e}")
                return gr.update(choices=[], value=[])

        def convert_files_to_faiss(auto_convert, selected_news_files, files, use_metadata, category_filter, topic_filter):
            """FAISS 변환 핸들러 (자동/수동 모드 지원)"""
            if auto_convert:
                # 자동 변환 모드: news_data 폴더 사용
                cat_filter = None if category_filter == "all" else category_filter
                top_filter = None if not topic_filter.strip() else topic_filter.strip()
                
                # generator를 처리하기 위해 yield from 사용
                yield from auto_convert_news_data_to_faiss(
                    selected_files=selected_news_files if selected_news_files else None,
                    use_metadata=use_metadata, 
                    category_filter=cat_filter, 
                    topic_filter=top_filter
                )
            else:
                # 수동 업로드 모드
                if not files:
                    yield None, "Please upload files or enable auto-convert mode."
                    return
                
                cat_filter = None if category_filter == "all" else category_filter
                top_filter = None if not topic_filter.strip() else topic_filter.strip()
                
                # generator를 처리하기 위해 yield from 사용
                yield from convert_to_faiss_indexes(files, use_metadata, cat_filter, top_filter)


        # 검색 버튼 클릭 이벤트 연결
        search_button.click(
            fn=search_with_metadata_filters,
            inputs=[query_input, use_date_filter, use_category_filter, use_topic_filter, category_select, topic_select, top_k],
            outputs=[results_output, date_range_display],
            show_progress=True
        )

        # 하이브리드 검색 버튼 클릭 이벤트 연결
        hybrid_search_button.click(
            fn=hybrid_search_with_answer,
            inputs=[query_input, use_date_filter, use_category_filter, use_topic_filter, 
                   category_select, topic_select, use_hybrid_search, generate_comprehensive_answer, top_k],
            outputs=[results_output, date_range_display],
            show_progress=True
        )

        # 필터 업데이트 버튼 클릭 이벤트
        update_filters_btn.click(
            fn=update_available_filters,
            outputs=[category_select, topic_select]
        )

        # 뉴스 파일 새로고침 버튼 클릭 이벤트
        refresh_news_files_btn.click(
            fn=refresh_news_files,
            outputs=[news_data_files]
        )

        # FAISS 변환 버튼 클릭 이벤트 (자동/수동 모드 지원)
        convert_to_faiss_btn.click(
            fn=convert_files_to_faiss,
            inputs=[auto_convert_news_data, news_data_files, faiss_upload_button, 
                   faiss_use_metadata, faiss_category_filter, faiss_topic_filter],
            outputs=[faiss_output, faiss_status_output],
            show_progress=True
        )


        # 모델 초기화 버튼 클릭 이벤트 연결
        init_model_btn.click(
            fn=init_model,
            inputs=[api_key, google_api_key, google_cse_id],
            outputs=[model_status],
        )

        # 인덱스 검색 정보
        index_info_button.click(
            update_index_info,
            outputs=[index_info_output]
        )

        # 파일 업로드 버튼 핸들러
        upload_button.change(
            fn=process_uploaded_files,
            inputs=[upload_button, exp_type, question_type],
            outputs=[file_output, status_output],
            show_progress=True
        )

        # 페이지 로드 시 뉴스 파일 목록 초기화
        demo.load(
            fn=refresh_news_files,
            outputs=[news_data_files]
        )

    return demo

def warmup_models():
    """모델 워밍업 - 첫 번째 요청 지연시간 감소"""
    global global_retriever, global_retriever_model, global_tokenizer, global_manager

    try:
        print("Warming up models...")

        # 더미 쿼리로 모델 워밍업
        dummy_query = "test query for warmup"

        if global_retriever and global_retriever_model and global_tokenizer:
            # 간단한 인코딩 테스트
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                # GPU 메모리 워밍업
                import time
                start_time = time.time()

                # 임베딩 모델 워밍업
                global_manager.embeddings.embed_query(dummy_query)

                print(f"Model warmup completed in {time.time() - start_time:.2f} seconds")

        print("✅ Models are warmed up and ready")

    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")

# ...existing code...

if __name__ == "__main__":
    import sys
    import os
    
    # 보안 설정 옵션 (기본값: 기존과 동일하게 외부 접근 허용)
    SECURE_MODE = os.getenv('SECURE_MODE', 'false').lower() == 'true'
    AUTH_USERNAME = os.getenv('AUTH_USERNAME', None)
    AUTH_PASSWORD = os.getenv('AUTH_PASSWORD', None)
    
    # 인증 설정
    auth_tuple = None
    if AUTH_USERNAME and AUTH_PASSWORD:
        auth_tuple = (AUTH_USERNAME, AUTH_PASSWORD)
        print(f"🔐 Authentication enabled for user: {AUTH_USERNAME}")
    
    server_port = int(sys.argv[1]) if len(sys.argv) > 1 else 7870
    
    if SECURE_MODE:
        print("🔒 Starting in SECURE MODE (localhost only)")
        print(f"   - Server will be accessible only on localhost:{server_port}")
        print(f"   - No external sharing enabled")
        
        demo = create_gradio_interface()
        demo.launch(
            share=False,  # 외부 공유 비활성화
            server_name="127.0.0.1",  # localhost만 허용
            server_port=server_port,
            debug=False,  # 보안을 위해 디버그 모드 비활성화
            auth=auth_tuple,  # 환경변수로 설정된 인증 사용
            show_error=False  # 에러 정보 숨김
        )
    else:
        # 기존과 동일한 설정 (외부 접근 허용)
        print(f"🌐 Starting server on port {server_port}...")
        print(f"   - External access enabled (same as before)")
        if auth_tuple:
            print(f"   - Authentication enabled for user: {auth_tuple[0]}")
        
        demo = create_gradio_interface()
        demo.launch(
            share=True,  # 공유 링크 활성화 (기존과 동일)
            server_name="0.0.0.0",  # 모든 IP에서 접근 허용 (기존과 동일)
            server_port=server_port,
            debug=True,  # 디버그 모드 활성화 (기존과 동일)
            auth=auth_tuple  # 환경변수로 설정된 인증 사용
        )

    # 실행 방법 CUDA_VISIBLE_DEVICES=8 python run.py (7861))