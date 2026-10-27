# Tài liệu tính năng KISBranche & Tổng quan tech stack AI Challenge HCMC 2026

> Nguồn tham khảo: [vkhanh2802/KISBranche](https://github.com/vkhanh2802/KISBranche) và các repo tương tự cho AI Challenge HCMC 2026.
> Ngày tổng hợp: 2026-10-25

---

## Phần 1 — Tính năng repo KISBranche

### 1.1. Tổng quan

KISBranche (tên trong README: **AIC 2026 KIS Retrieval**) là hệ thống tìm keyframe video (KIS — Known-Item Search) cho cuộc thi AI Challenge HCMC 2026. Người dùng nhập truy vấn thô (raw query) bằng tiếng Việt, hệ thống trả về **top-20 keyframe** phù hợp nhất qua giao diện web FastAPI.

Kiến trúc pipeline: **đa luồng retrieval (3 luồng) → fusion + boost → rerank bằng VLM → hiển thị kết quả**.

### 1.2. Các tính năng chính

#### A. Truy vấn & tiền xử lý truy vấn
- **Gemini Query Structuring**: dùng Gemini API chuyển truy vấn thô của người dùng thành truy vấn có cấu trúc (`structure_query.py`), hỗ trợ truy vấn tiếng Việt.
- **Gemini OCR (tùy chọn)**: bóc tách văn bản trên keyframe (`gemini_ocr.py`), chạy trước khi build chỉ mục caption.
- Debug truy vấn độc lập bằng `structure_query.py` + notebook `runbranch.ipynb`.

#### B. Ba luồng retrieval (multi-stream retrieval)
| Luồng | Công nghệ | Dữ liệu |
|---|---|---|
| Tìm kiếm thị giác | **SigLIP** + vector DB **Milvus** | Embedding keyframe (`.npy`, shape `(N, 768)`) |
| Tìm kiếm từ khóa | **BM25** trên caption | Caption sinh bởi BLIP (+ BLIP VQA) |
| Tìm kiếm ngữ nghĩa | **SentenceTransformer** (MiniLM, 384 chiều) trên caption, chỉ mục **FAISS** | Caption |

#### C. Boost & Rerank
- **Object metadata boost (tùy chọn)**: tăng hạng kết quả khớp metadata vật thể từ object detection (**SSDLite MobileNetV3**, 80 nhãn COCO, map sang schema Open Images; `object_filter.py`, `apply_object_filter.py`, `object_label_map.json`).
- **VLM Reranker**: **Qwen2.5-VL** (4-bit, ~6 GB VRAM) rerank top-20 kết quả (`vlm_reranker.py`, `vlm_scoring.py`). Có thể tắt bằng biến môi trường `KIS_DISABLE_VLM=1` khi chạy máy không GPU.

#### D. Xây dựng dữ liệu & chỉ mục (offline pipeline)
- `prepare_video_data.py` + notebook **`prepare_videos_cpu.ipynb`**: chuẩn bị dữ liệu video mới trên server **không cần GPU** (keyframe sampling: frame 0 + khoảng cách ngẫu nhiên 80–160 frame, seed theo video_id).
- `caption_generator.py`: sinh caption bằng BLIP; `build_caption_index.py`: build chỉ mục FAISS cho caption.
- `build_milvus.py build|rebuild`: tạo/tái tạo collection `clip_keyframes` trong Milvus.
- **Milvus qua Docker Compose** (etcd + minio + standalone, port 19530).
- `migrate_caption_artifacts.py`, `audit_artifact_formats.py`: migration & kiểm tra định dạng artifact.

#### E. Web app
- `web_app.py`: **FastAPI + Uvicorn** (1 worker), chạy tại `http://127.0.0.1:8000`, giao diện trong thư mục `web/`.
- Quy trình hằng ngày: activate env → `docker compose up -d` → set `GEMINI_API_KEY` → chạy app (không cần rebuild nếu dữ liệu không đổi).

#### F. Reproducibility & kiểm thử
- **Model pinning**: các model ghim theo revision bất biến trong `model_manifest.json`.
- **Caption artifact manifest SHA-256**: `CaptionRetriever` từ chối artifact thiếu/sai checksum thay vì âm thầm tìm trên dữ liệu cũ.
- **~13 file test** (`test_*.py`) chạy bằng `python -m unittest discover`.
- Troubleshooting đầy đủ trong README: thiếu API key, mất kết nối Milvus 19530, collection sai metadata, thiếu caption/index, lỗi CUDA/OOM.

### 1.3. Tech stack KISBranche (tóm tắt)

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ / môi trường | Python 3.11 (conda `focusguard_ai`), PyTorch 2.7.1 / CUDA 12.8 (tùy chọn) |
| Vector DB | Milvus (Docker Compose), FAISS |
| Embedding thị giác | SigLIP |
| Caption | BLIP + BLIP VQA |
| Embedding văn bản | SentenceTransformer MiniLM (384-d) |
| Từ khóa | BM25 |
| Object detection | SSDLite MobileNetV3 (COCO 80) |
| Rerank | Qwen2.5-VL (4-bit, ~6 GB VRAM, tắt được) |
| LLM / OCR đám mây | Gemini API (query structuring + OCR) |
| Backend / UI | FastAPI + Uvicorn |
| Kiểm thử | unittest |

> ⚠️ Lưu ý từ README: clone repo **chưa đủ để chạy** — cần các thư mục dữ liệu `data/`, `mapping/`, `clip-features-32/` (tối thiểu) được cung cấp ngoài (Drive/HuggingFace).

---

## Phần 2 — Tổng hợp tech stack các repo AI Challenge HCMC 2026 tương tự

### 2.1. Các repo tham khảo

| Repo | Mô tả ngắn |
|---|---|
| [pgtomvn/AIC2026](https://github.com/pgtomvn/AIC2026) | KIS + QA + TRAKE, FastAPI + Streamlit, Qdrant + Meilisearch |
| [ledinhminhquan/Core-Vision-AIC2026](https://github.com/ledinhminhquan/Core-Vision-AIC2026) | KIS/QA/TRAKE/AVS, FAISS + BM25, cross-encoder rerank |
| [ph4mHoa/aic2026-video-retrieval](https://github.com/ph4mHoa/aic2026-video-retrieval) | Truy vấn text/voice/sketch, RRF fusion, không dùng vector DB (numpy mmap) |
| [risc97/aic26-backend](https://github.com/risc97/aic26-backend) (+ frontend, preprocessing) | Hệ TRISC tách 3 repo: FastAPI + SQLite + turbovec; model C2LIP, SigLIP2, DINOv3, GTE, OWLv2; tìm keyframe/transcript/OCR/object/temporal/image-similarity |
| [ThuyHaLE/SceneSeek](https://github.com/ThuyHaLE/SceneSeek) | Jina CLIP v2 + FAISS (HNSW/FlatIP) + BM25 (pyvi), RRF fusion, FastAPI + React, chạy CPU |
| [vkhanh2802/KISBranche](https://github.com/vkhanh2802/KISBranche) | (repo chính của tài liệu này) |

### 2.2. Các điểm chung về tech stack

**1. Embedding thị giác đa phương thức — CLIP-family là chuẩn de-facto**
- **SigLIP / SigLIP-2** là mô hình được dùng nhiều nhất (KISBranche, pgtomvn, Core-Vision, ph4mHoa); các biến thể khác: Jina CLIP v2, MetaCLIP 2, OpenCLIP của ban tổ chức, CLIP32.
- Nhiều đội giữ **nhiều bộ embedding song song** (ví dụ 768-D + 512-D) và fuse kết quả.

**2. Hybrid retrieval: dense (vector) + sparse (BM25/FTS)**
- Tất cả các đội đều kết hợp tìm kiếm vector với tìm kiếm từ khóa trên **văn bản OCR và/hoặc ASR** — không đội nào chỉ dùng một luồng.
- Text search engine: BM25 tự quản/lib, SQLite FTS5, Meilisearch.

**3. Vector store: chia làm 2 trường phái**
- **FAISS** (Core-Vision, SceneSeek, KISBranche cho caption) — phổ biến nhất.
- **Milvus** (KISBranche) hoặc **Qdrant** (pgtomvn) khi cần vector DB "thật".
- ph4mHoa thậm chí bỏ hẳn vector DB: inner-product chính xác trên numpy memory-mapped matrix.

**4. OCR + ASR là bắt buộc**
- OCR: **PaddleOCR** (pgtomvn, ph4mHoa), VietOCR, Vintern-1B, hoặc Gemini OCR (KISBranche).
- ASR: **faster-whisper** (pgtomvn, ph4mHoa), PhoWhisper (Core-Vision) — ưu tiên model tiếng Việt.

**5. Fusion bằng RRF/Rank-fusion + rerank đa tầng**
- **Reciprocal Rank Fusion (RRF)** hoặc weighted-RRF là phương pháp fuse top-k phổ biến nhất (pgtomvn, ph4mHoa).
- Rerank tầng sau: **VLM/Qwen (Qwen2.5-VL, Qwen3-VL-Reranker)**, cross-encoder (BLIP-2 ITM), hoặc aggregation theo shot/cửa sổ thời gian.

**6. LLM dùng cho xử lý truy vấn, không cho retrieval trực tiếp**
- **Gemini** gần như là lựa chọn chuẩn cho query structuring/translation/expansion (KISBranche, Core-Vision, pgtomvn); các đội còn lại dịch truy vấn cục bộ bằng **MarianMT** hoặc rule-based.
- VLM (Gemini/Qwen) cho Visual QA và rerank.

**7. Backend & UI**
- **Python + FastAPI** là backend thống trị (KISBranche, pgtomvn, ph4mHoa); **Streamlit** là UI nhanh phổ biến (pgtomvn, Core-Vision).
- Kiểu kiến trúc 1 repo (monolith pipeline) như KISBranche, hoặc tách backend/frontend/preprocessing như risc97.

**8. RAG-style reproducibility & engineering**
- Model pinning theo revision, manifest checksum artifact (KISBranche), Pydantic contract + JSON Schema hash (ph4mHoa), CI GitHub Actions với test suite lớn (538–970 tests), Docker Compose cho hạ tầng (Milvus etcd/minio).
- Keyframe sampling được benchmark kỹ (grid 250 ms vs khoảng cách ngẫu nhiên — ph4mHào nâng recall cap từ ~20% lên >90% nhờ chọn lưới sampling đúng).

### 2.3. Bảng so nhanh

| | KISBranche | pgtomvn | Core-Vision | ph4mHoa | risc97 | SceneSeek |
|---|---|---|---|---|---|---|
| Embedding | SigLIP | SigLIP2 + OpenCLIP | SigLIP-2, MetaCLIP 2, Qwen3-VL-Embed | SigLIP-2/CLIP | SigLIP2, C2LIP, DINOv3, GTE | Jina CLIP v2, vietnamese-embedding |
| Vector store | Milvus + FAISS | Qdrant | FAISS | numpy mmap | turbovec + SQLite | FAISS (HNSW/FlatIP) |
| Từ khóa | BM25 | Meilisearch | BM25 | SQLite FTS5 | fuzzy match OCR/transcript | BM25 (pyvi) |
| OCR | Gemini | PaddleOCR, VietOCR | Vintern-1B | PaddleOCR | (repo preprocessing) | — |
| ASR | — | faster-whisper | PhoWhisper | faster-whisper | (repo preprocessing) | — |
| Fusion | boost + rerank | weighted RRF | ensemble + boosts | RRF + temporal rerank | — | RRF |
| Rerank | Qwen2.5-VL | retrieval module | BLIP-2 ITM, Qwen3-VL | temporal | — | /refine (TODO) |
| LLM | Gemini | Gemini + Ollama Qwen | Gemini | MarianMT (không LLM) | — | — (dự kiến) |
| Backend | FastAPI | FastAPI + Streamlit | Streamlit + HTTP service | Pydantic contract | FastAPI + React/TS | FastAPI + React |
| Chạy không GPU | ✅ (KIS_DISABLE_VLM) | profile 4 GiB | ✅ | ✅ | CUDA khuyến nghị | ✅ (CPU) |

### 2.4. Kết luận

Tech stack "chuẩn" của một đội AI Challenge HCMC 2026 cho bài toán video retrieval: **SigLIP(-2) làm embedding thị giác + PaddleOCR/VietOCR cho OCR + faster-whisper/PhoWhisper cho ASR + BM25/FTS cho từ khóa + FAISS/Milvus/Qdrant làm vector store + RRF fusion + VLM (Qwen/Gemini) rerank + Gemini cho xử lý truy vấn tiếng Việt + FastAPI/Streamlit làm backend/UI + Docker Compose cho hạ tầng**. KISBranche nằm đúng trong khuôn khổ này, điểm khác biệt là dùng **Milvus làm vector DB chính, Gemini cho cả query structuring lẫn OCR, và có chế độ chạy CPU-only**.
