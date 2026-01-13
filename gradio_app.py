#!/usr/bin/env python3
"""
Gradio Web Interface for Multimodal RAG Search

Features:
  - Multi-collection search support
  - Text/Image/Hybrid search modes
  - LLM-powered answer generation (OpenAI/Anthropic)
  - Topic filtering
  - Interactive UI with real-time search

Usage:
  python gradio_app.py [PORT]
  
  Default port: 7860
  Example: python gradio_app.py 7870
"""

import gradio as gr
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from project modules
from pymilvus import MilvusClient

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available - image search disabled")

import numpy as np
import json

# Import AnswerGenerator from rag_core
from rag_core import AnswerGenerator


class GradioSearcher:
    """Gradio용 간소화된 검색 클래스 (run_search.py의 MultimodalSearcher 기반)"""
    
    def __init__(self, db_file: str, collections: List[str], text_model: str = "jhgan/ko-sroberta-multitask"):
        self.collections = collections if isinstance(collections, list) else [collections]
        self.client = MilvusClient(db_file)
        self.db_file = db_file
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name=text_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if CLIP_AVAILABLE:
            self.device = device
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP model loaded for image search")
    
    def search_by_text(self, query: str, top_k: int = 5, topic: str = None) -> List[Dict]:
        """텍스트 검색"""
        filter_expr = f'topic == "{topic}"' if topic and topic != "all" else None
        query_vector = self.text_embeddings.embed_query(query)
        
        all_results = []
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
                hit['_collection'] = coll
                all_results.append(hit)
        
        all_results.sort(key=lambda x: x.get('distance', 0), reverse=True)
        return all_results[:top_k]
    
    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """이미지 검색"""
        if not CLIP_AVAILABLE:
            logger.error("CLIP not available for image search")
            return []
        
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            img_features = self.clip_model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        
        query_vec = img_features.cpu().numpy().flatten().tolist()
        
        all_results = []
        for coll in self.collections:
            if not self.client.has_collection(coll):
                continue
            
            # 이미지가 있는 문서만 검색
            results = self.client.search(
                collection_name=coll,
                data=[self.text_embeddings.embed_query("화재")],
                limit=500,
                filter="has_image == true",
                output_fields=["doc_id", "title", "content", "date", "source", "has_image", "image_path", "image_embedding", "category", "topic"]
            )
            
            for r in (results[0] if results else []):
                entity = r.get('entity', {})
                emb_str = entity.get('image_embedding', '[]')
                if not emb_str or emb_str == '[]':
                    continue
                
                stored_vec = json.loads(emb_str)
                sim = np.dot(query_vec, stored_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
                r['distance'] = float(sim)
                r['_collection'] = coll
                all_results.append(r)
        
        all_results.sort(key=lambda x: x['distance'], reverse=True)
        return all_results[:top_k]


# Global instances
searcher = None


def init_app(db_file: str, collections: str):
    """앱 초기화"""
    global searcher
    
    try:
        colls = [c.strip() for c in collections.split(',')]
        searcher = GradioSearcher(db_file=db_file, collections=colls)
        logger.info(f"✅ Initialized: {db_file} | Collections: {colls}")
        return f"✅ Initialized: {db_file}\n📁 Collections: {', '.join(colls)}"
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return f"❌ Initialization failed: {str(e)}"


def format_results(results: List[Dict]) -> str:
    """검색 결과를 Markdown 형식으로 포맷"""
    if not results:
        return "검색 결과가 없습니다."
    
    output = []
    for i, r in enumerate(results, 1):
        entity = r.get('entity', {})
        output.append(f"### [{i}] {entity.get('title', 'N/A')}")
        output.append(f"**Score:** {r.get('distance', 0):.4f} | **Collection:** `{r.get('_collection', '')}`")
        output.append(f"📅 {entity.get('date', 'N/A')} | 🏷️ {entity.get('topic', 'N/A')} | 📰 {entity.get('source', 'N/A')}")
        
        content = entity.get('content', '')[:300].replace('\n', ' ')
        output.append(f"\n{content}...\n")
        output.append("---\n")
    
    return "\n".join(output)


def search_fn(query: str, mode: str, image, top_k: int, topic: str, generate_answer: bool, llm: str):
    """검색 및 답변 생성"""
    if not searcher:
        return "⚠️ 먼저 DB를 초기화하세요 (Settings 탭에서 Initialize 버튼 클릭)", ""
    
    results = []
    
    try:
        # 검색 모드에 따라 실행
        if mode == "text" and query:
            results = searcher.search_by_text(query, top_k=top_k, topic=topic if topic != "all" else None)
        
        elif mode == "image" and image:
            results = searcher.search_by_image(image, top_k=top_k)
        
        elif mode == "hybrid" and query and image:
            text_results = searcher.search_by_text(query, top_k=top_k*2, topic=topic if topic != "all" else None)
            img_results = searcher.search_by_image(image, top_k=top_k*2)
            
            # 결과 결합
            combined = {}
            for r in text_results:
                doc_id = r.get('entity', {}).get('doc_id')
                if doc_id:
                    combined[doc_id] = r
            
            for r in img_results:
                doc_id = r.get('entity', {}).get('doc_id')
                if doc_id and doc_id in combined:
                    combined[doc_id]['distance'] = (combined[doc_id]['distance'] + r['distance']) / 2
                elif doc_id:
                    combined[doc_id] = r
            
            results = sorted(combined.values(), key=lambda x: x['distance'], reverse=True)[:top_k]
        
        else:
            return "⚠️ 검색 모드에 맞는 입력을 제공하세요.", ""
        
        # 검색 결과 포맷팅
        formatted = format_results(results)
        
        # 답변 생성 (선택사항)
        answer = ""
        if generate_answer and results and query:
            logger.info(f"Generating answer with {llm}...")
            gen = AnswerGenerator(llm=llm)
            answer = gen.generate(query, results)
        
        return formatted, answer
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ 검색 오류: {str(e)}", ""


def create_interface():
    """Gradio 인터페이스 생성"""
    with gr.Blocks(title="🔍 Multimodal RAG Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Multimodal RAG Search System
        
        **Features:**
        - 📝 Text Search: 텍스트 질의로 검색
        - 🖼️ Image Search: 이미지로 유사한 뉴스 검색
        - 🎯 Hybrid Search: 텍스트와 이미지를 결합한 검색
        - 🤖 Answer Generation: LLM을 활용한 답변 생성
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                db_file = gr.Textbox(label="DB File Path", value="./db/fire.db", placeholder="./db/fire.db")
                collections = gr.Textbox(
                    label="Collections (comma-separated)",
                    value="fire_multimodal_demo,disaster_manual",
                    placeholder="fire_news,disaster_manual"
                )
                init_btn = gr.Button("🚀 Initialize", variant="primary")
                init_status = gr.Textbox(label="Status", interactive=False, lines=3)
                
                gr.Markdown("### 🔧 Search Options")
                mode = gr.Radio(["text", "image", "hybrid"], label="Search Mode", value="text")
                top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K Results")
                topic = gr.Dropdown(
                    ["all", "fire", "earthquake", "flood", "typhoon", "heatwave", "coldwave", "chemical", "wildfire", "collapse"],
                    label="Topic Filter",
                    value="all"
                )
                
                gr.Markdown("### 🤖 Answer Generation")
                generate_answer = gr.Checkbox(label="Generate Answer with LLM", value=True)
                llm = gr.Radio(["openai", "anthropic"], label="LLM Provider", value="openai")
                gr.Markdown("*Requires API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)*")

            with gr.Column(scale=2):
                gr.Markdown("### 📝 Query Input")
                query = gr.Textbox(
                    label="Text Query",
                    placeholder="예: 화재 발생 시 대피 방법은?",
                    lines=2
                )
                image = gr.Image(label="Image Query (Optional for image/hybrid mode)", type="filepath")
                search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                
                gr.Examples(
                    examples=[
                        ["화재 발생 시 대피 방법은?"],
                        ["지진이 발생하면 어떻게 해야 하나요?"],
                        ["폭염 시 건강 관리 방법"],
                        ["화학사고 발생 시 행동요령"],
                        ["산불이 주거지역에 접근할 때 대처법"]
                    ],
                    inputs=[query]
                )
                
                gr.Markdown("### 📊 Search Results")
                results_output = gr.Markdown(label="Search Results")
                
                gr.Markdown("### 💬 Generated Answer")
                answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)

        # Event handlers
        init_btn.click(init_app, inputs=[db_file, collections], outputs=[init_status])
        
        search_btn.click(
            search_fn,
            inputs=[query, mode, image, top_k, topic, generate_answer, llm],
            outputs=[results_output, answer_output]
        )
        
        query.submit(
            search_fn,
            inputs=[query, mode, image, top_k, topic, generate_answer, llm],
            outputs=[results_output, answer_output]
        )
    
    return demo


if __name__ == "__main__":
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    
    logger.info(f"Starting Gradio app on port {port}...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True
    )
