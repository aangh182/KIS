# Plan & Tổng hợp: Hệ thống Video Retrieval — Known-Item Search (Dataset AI Challenge HCMC 2024)

> Tài liệu dành cho người mới tìm hiểu về video retrieval. Bao gồm:
> (1) thể lệ và hệ quả thiết kế, (2) dataset AIC 2024 có gì, (3) giải thích chi tiết pipeline cơ bản của một hệ thống retrieval, (4) pipeline chi tiết của các repo tham khảo năm 2024 và 2025–2026, (5) bảng so sánh rõ ràng, (6) plan đề xuất cho đội mình.
>
> Tài liệu liên quan: `docs/Plan-KIS-Final-Round.md` (quyết định kiến trúc đã chốt sau grill-me), `docs/Schema-Reference.md` (schema SQLite đã chốt), `docs/KISBranche-Features.md` (tính năng repo KISBranche + tech stack 2026).
> Ngày tổng hợp: 2026-10-26.

---

## Phần 1 — Thể lệ vòng chung kết và hệ quả thiết kế

### 1.1. Ba dạng truy vấn

| Dạng | Đầu vào | Thời gian | Đầu ra |
|---|---|---|---|
| **Textual KIS** | Mô tả ngôn ngữ tự nhiên của một sự kiện, **cung cấp dần dần** theo thời gian (nhiều ý, nhiều câu) | 5 phút/query | Video + thời điểm chính xác |
| **Video KIS** | Xem đoạn video ≤ 20 giây trích từ dataset (không được ghi hình lại bằng thiết bị điện tử) → phải **mô tả lại bằng chữ hoặc vẽ** | 4 phút/query | Video + thời điểm chính xác |
| **Q&A** | Mô tả sự kiện (cung cấp dần) + câu hỏi có sẵn ngay từ đầu; đáp án là chuỗi ký tự "0"–"9" | 5 phút/query | Video + đáp án số |

### 1.2. Công thức điểm và hệ quả

`Score = (50 − 10×ws) + 50×(1 − t/tL)` với `ws` = số lần nộp sai, `t` = thời điểm nộp đúng.

Hệ quả trực tiếp cho thiết kế:
1. **Tốc độ search là điểm số** — mỗi lần search thêm là mất điểm thời gian. Kết quả phải về < 2 giây.
2. **Nộp sai mất ít (10đ) so với phần thời gian (tối đa 50đ)** → chiến lược "nộp sớm khi tin cậy ≥ ~60%" có kỳ vọng dương. UI phải hiển thị "điểm được nếu nộp đúng ngay lúc này".
3. **Mô tả dần dần** → UI phải cho **append** text vào query và search lại (không gõ lại từ đầu), có khay ghim ứng viên giữa các lần search.
4. **Video KIS** → người dùng phải gõ mô tả bằng tay → checklist thuộc tính (bối cảnh/trang phục/chữ trên màn hình/hành động) giúp gõ nhanh; **chữ trên màn hình (OCR) là kênh mạnh nhất** vì dễ mô tả chính xác.
5. **Q&A** → tìm được frame rồi còn phải **đọc được con số** trên frame → UI cần zoom + scrub frame kề.

---

## Phần 2 — Dataset AI Challenge HCMC 2024

- **Nguồn**: video tin tức YouTube, ~800+ video (~292 giờ), chia theo batch `L01`–`L3x`, đặt tên `Lxx_Vyyy.mp4`. Workspace hiện có thư mục `Dataset/Video/` (ví dụ `L21_V024.mp4`).
- **BTC cung cấp sẵn** (nhiều đội 2024 chỉ cần tải về và nạp vào index là có hệ thống chạy):
  - `Keyframes/` + `map-keyframes/`: ảnh keyframe trích sẵn + CSV ánh xạ `n, pts_time, fps, frame_idx` (keyframe → thời điểm).
  - `Clip-features/`: embedding CLIP cho từng keyframe, nhiều biến thể (`clip_features_l14`, `b16`, `b32` — file `.npy` mỗi video, shape `(N, dim)`).
  - `Media-info/`: JSON caption/thể loại mỗi video.
- **Lưu ý quan trọng**: clip-features của BTC (CLIP ViT-L/14) chỉ là **baseline** — các đội top đều tự embedding bằng model tốt hơn (SigLIP, DFN5B, bigG/14) và tăng score rõ rệt. Ngoài ra video trong dataset có thể **VFR (variable frame rate)** — tuyệt đối không tính thời gian bằng `frame_idx / fps`, phải dùng `pts_time` từ CSV map-keyframes (bài học của repo ph4mHoa, đã ghi trong `docs/Schema-Reference.md`).

---

## Phần 3 — Pipeline cơ bản của một hệ thống video retrieval (giải thích cho người mới)

Một hệ thống retrieval cho bài KIS gồm **2 giai đoạn tách biệt**: **offline** (chuẩn bị dữ liệu, chạy một lần trước thi) và **runtime** (chạy liên tục trong giờ thi, mỗi lần search).

### 3.1. Giai đoạn offline — xây dựng "cuốn từ điển"

Mục tiêu: biến ~800 video thành các **chỉ mục (index)** tra được trong mili-giây.

```
Video (.mp4)
  │
  ├─(1) Keyframe extraction ──→ Ảnh đại diện mỗi khoảnh khắc
  │
  ├─(2) Embedding ────────────→ Vector số đặc trưng ngữ nghĩa thị giác
  │
  ├─(3) OCR ──────────────────→ Chữ xuất hiện trong frame (banner, biển hiệu, phụ đề)
  │
  ├─(4) ASR ──────────────────→ Lời thoại trong video (transcript)
  │
  ├─(5) Metadata ─────────────→ Thời gian, video_id, đường dẫn ảnh
  │
  └─→ Lưu vào: FAISS (vector) + SQLite (metadata + FTS5 full-text) + manifest checksum
```

**(1) Keyframe extraction.** Không thể index toàn bộ frame (292 giờ × 25fps ≈ 26 triệu frame). Chọn các "khoảnh khắc đại diện": cách rẻ nhất là lấy 1 frame mỗi giây (uniform sampling); cách tốt hơn là **Shot Boundary Detection** (PySceneDetect/Autoshot/TransNetV2) — cắt video theo cảnh, lấy 1–n frame mỗi cảnh, recall cao hơn vì không bỏ sót cảnh ngắn. Các đội 2024 được BTC trích sẵn keyframe; các đội 2026 tự trích. **Bài học đo đạc** (ph4mHoa, 2026): lưới sampling sai có thể giới hạn recall tối đa ở ~20% — phải benchmark sampling bằng số đo recall, không chọn bằng cảm tính.

**(2) Embedding (image → vector).** Mỗi keyframe đưa qua model đa phương thức (CLIP-family: CLIP, SigLIP, SigLIP-2, Jina CLIP v2...) thành vector 384–1152 chiều. Tính chất cốt lõi: **ảnh và text cùng một không gian vector** — mô tả "cảnh chống lều ở bãi biển" cho vector gần với vector của ảnh chứa cảnh đó. Dot-product/cosine giữa query vector và mọi keyframe vector = độ tương đồng. Đây là "luồng thị giác" của pipeline. Embedding là phép tính nặng (cần GPU) → **precompute offline một lần**, giờ thi chỉ encode text query (~100ms trên CPU).

**(3) OCR.** Đọc chữ trong keyframe (PaddleOCR/EasyOCR/VietOCR/Gemini OCR). Là kênh **rất mạnh** cho cuộc thi này vì video tin tức đầy chữ: tên người, địa danh, banner sự kiện, phụ đề. Mô tả "người dẫn chuyện đứng trước banner **DIỄN ĐÀN Mekong 2023**" khớp gần như duy nhất qua OCR, trong khi embedding thị giác có thể trả về mọi cảnh studio.

**(4) ASR.** Chuyển lời thoại thành text (faster-whisper/PhoWhisper — model tiếng Việt). Bắt theo **thời gian**, không theo keyframe (một đoạn transcript 3 giây map về keyframe tại thời điểm đó). Phủ nội dung không hiện lên màn hình: phát biểu, phỏng vấn.

**(5) Metadata & toàn vẹn.** Mỗi keyframe cần danh tính đầy đủ: `(video_id, kf_n, frame_idx, t_ms, image_path)`. Chuẩn hóa **một lần** lúc ingest (bài học SceneSeek), checksum manifest để tái tạo được.

### 3.2. Giai đoạn runtime — mỗi lần bấm search

```
Query text (tiếng Việt, gõ dần)
  │
  ├─(A) Query processing ─→ [tùy chọn, LLM] tách thuộc tính, dịch/expand sang EN
  │
  ├─(B) Tìm song song trên 3 luồng:
  │     • Luồng VISUAL: encode query bằng CLIP/SigLIP text tower → tìm FAISS top-k
  │     • Luồng OCR:  BM25/SQLite FTS5 trên kho chữ đã OCR
  │     • Luồng ASR:   BM25/FTS5 trên transcript
  │
  ├─(C) FUSION: gộp 3 bảng xếp hạng thành 1 (RRF — xem 3.3)
  │
  ├─(D) RERANK (tùy chọn): dùng model/LLM mạnh hơn chấm lại top-20
  │
  └─(E) Render: grid keyframe kèm điểm số + nguồn match, để operator tự xác nhận
```

**(A) Query processing.** Query tiếng Việt nhưng nhiều model CLIP hiểu tiếng Anh tốt hơn. Các đội: dịch máy (MarianMT/googletrans) hoặc để **LLM (Gemini) cấu trúc hóa** query — tách địa điểm/nhân vật/vật thể/chữ trên màn hình, dịch, sinh từ khóa cho từng luồng. Phải có **cache + timeout ngắn (2s) + fallback dùng nguyên văn** — Gemini lag không được phép làm treo search.

**(B) Tìm kiếm song song.** Mỗi luồng trả top-k riêng. Luồng vector dùng FAISS (thư viện của Meta, tìm k láng giềng gần nhất trong hàng triệu vector, vài ms) hoặc vector DB thật (Qdrant/Milvus — cần Docker, ops nặng hơn). Luồng text dùng BM25/SQLite FTS5 (full-text search, khớp từ khóa chính xác — thứ embedding hay bỏ sót tên riêng, con số).

**(C) Fusion — tại sao cần và RRF hoạt động thế nào.** Điểm số của 3 luồng **không cùng thang** (cosine 0.7 của luồng visual và BM25 score 12.3 của luồng OCR không so sánh được). **Reciprocal Rank Fusion** giải quyết bằng cách chỉ dùng **vị trí hạng**: với keyframe X, `RRF(X) = Σ_channels w_c / (k + rank_c(X))`, k≈60. Keyframe hạng 2 luồng visual + hạng 5 luồng OCR sẽ nổi lên trên keyframe chỉ hạng 1 một luồng — đúng ý nghĩa: nhiều nguồn xác nhận = tin cậy hơn. Trọng số `w_c` tune trên eval set.

**(D) Rerank.** Fusion chỉ nhìn hạng; rerank nhìn lại **nội dung**: VLM (Qwen2.5-VL local — cần GPU) hoặc LLM đám mây (Gemini) nhận "query + 20 thumbnail" và chấm lại. Chỉ tăng chất lượng nếu recall tầng trước đã đủ tốt; phải là bước tùy chọn, tắt được khi lỗi.

**(E) Render có giải thích nguồn.** Hiển thị keyframe khớp vì *luồng nào* (visual/OCR/ASR) — operator biết nên tin kết quả ở mức nào khi quyết định nộp bài.

### 3.3. Vì sao kiến trúc này là "chuẩn chung" của mọi đội

Không đội nào thắng chỉ nhờ một model. Điểm số đến từ: **(i)** embedding tốt (SigLIP-2 > CLIP-L14 BTC), **(ii)** phủ đủ kênh (visual + OCR + ASR) vì mỗi kênh giải quyết dạng truy vấn khác nhau, **(iii)** fusion hợp lý, **(iv)** UI tốt — phần lớn thời gian giờ thi là thời gian *người* đọc kết quả và quyết định nộp, không phải thời gian máy tính.

---

## Phần 4 — Pipeline chi tiết các repo tham khảo

### 4.1. Các repo AI Challenge HCMC 2024 (năm của dataset)

#### a) VinhAlth/Pumpkin_AIC_2024 — Team Pumpkin (SIU), Top-10 chung kết
- **Offline**: tự trích lại keyframe bằng Shot Boundary Detection (PySceneDetect + Autoshot fine-tuned trên AIC 2022), khử near-duplicate bằng perceptual hash; VAD (Pyannote) + ASR Wav2Vec2 fine-tuned VLSP (tiếng Việt).
- **Embedding**: SigLIP + CLIP DFN5B (DataComp) — cho phép chọn model lúc query.
- **Index**: Qdrant + **binary quantization** (nén vector 32×, search ~27× nhanh, ~50ms/query).
- **Điểm mạnh đặc trưng — Temporal-aware scoring**: cộng/nhân (additive/multiplicative) cosine score trên cửa sổ thời gian Δ=1500s để xử lý mô tả gồm nhiều cảnh liên tiếp; rerank theo story-segment, filter theo màu, perceptual hash, metadata, transcript.
- **UI**: Flask + Gunicorn, thumbnail AVIF, video AV1, vocal query, phím tắt.
- **Kết quả**: 94% accuracy vòng loại; ~800ms cho top-200.
- **Bài học**: SBD tốt + temporal aggregation + quantization = vừa nhanh vừa chính xác; đây là paper chính thức của team tại AI Challenge 2024.

#### b) chisngooo/Multimodal-Video-Retrieval-Engine — Team NaiveNotNice
- Dùng keyframes + clip-features BTC; thêm OCR, ASR, image captioning, object & color detection.
- **Multi-CLIP ensemble**: chạy gần như cả "thế giới CLIP" qua open_clip (ViT-B/32, L/14, H/14 Apple, bigG/14 LAION, QuickGELu...), **mỗi biến thể một FAISS index riêng**, query song song nhiều model rồi hợp nhất; bật/tắt model qua config để giảm thời gian khởi động.
- **Elasticsearch 7.x** cho text search theo OCR/ASR/object detection.
- FastAPI + Jinja2; dịch query Việt↔Anh (deep-translator, pyvi, underthesea).
- **Bài học**: chạy nhiều model embedding song song và hợp nhất là cách tăng recall rẻ nhất, nhưng trả giá bằng index size và thời gian init.

#### c) MinLee0210/evento — Team AIO_TOP10
- Tự trích keyframe bằng **TransNetV2 + adaptive clustering**; sinh caption bằng **BLIP2**; tạo synthetic data từ mô tả để bù semantic gap.
- **Một trong số ít đội 2024 dùng LLM**: **Gemini API** cho query expansion + keyword extraction; Redis cache "share search" cho query trùng.
- Hạ tầng MySQL + MongoDB + Redis; **FastAPI + Streamlit**; API fuzzy-match query với OCR text.
- **Bài học**: Gemini query expansion đã chứng minh giá trị từ 2024; caption BLIP2 là kênh trung gian giữa visual và text thuần.

#### d) LoylP/AIC2024 — ArtemisSearch
- Tự extract 1 frame/giây; embedding **ViT-H/14 + BEiT3** (BEiT3 là kiến trúc hiếm); **Milvus** làm vector store; EasyOCR làm filter; **tách query dài thành chuỗi câu con** để tìm chuỗi sự kiện (hướng temporal tương tự Pumpkin).
- FastAPI + React + Elasticsearch.
- **Bài học**: temporal decomposition của query nhiều ý là kỹ thuật đặc trưng mùa 2024.

#### e) BeQuang/AIC2024 + xlinh2301/aic_backend (cùng đội)
- Keyframes BTC + OCR + ASR + object detection; CLIP; **FAISS** (faiss-gpu); **Elasticsearch qua Docker kèm backup file + fuzzy fallback khi ES sập**; FastAPI + React.
- **Bài học**: quan tâm độ bền — mọi dịch vụ ngoài (ES, Gemini) phải có fallback; đúng tinh thần "mọi tính năng có fallback" đã chốt trong plan của mình.

#### f) lnkhang1305/AIC2024_Video_Retrieval
- "Recipe chung" thuần của mùa 2024: clip-features L14/B16/B32 của BTC nạp vào **Qdrant (Docker, 3 collection)**, Flask + server tĩnh host ảnh keyframe.
- **Bài học**: là baseline tối giản — nạp npy có sẵn là có hệ thống; dùng làm điểm so sánh khi benchmark model của mình.

#### g) JakeClark-chan/Video-Retrieval-for-AIC-2024
- YOLOv10x object detection + dominant color (K-Means) + CLIP; Qdrant + SQLite; **Streamlit có Drawable Canvas (vẽ bounding box tìm frame) + filter màu/object**.
- **Bài học**: tương tác trực quan (sketch/filter) là hướng hiếm nhưng phù hợp Video KIS.

#### h) QuanHoangNgoc/...-AIC-2024-HCMC-Final (vào chung kết)
- BEiT3 + googletrans + FAISS; "two-phase / element-centered search" — tách query thành thành phần thị giác rồi hợp nhất; tự thiết kế UI.
- **Bài học**: đội vào chung kết không cần stack đồ sộ — cần pipeline có kiểm soát + UI tốt.

#### i) Khác đáng chú ý
- **tb-tian/AIChallenge2024**: pipeline offline sạch (keyframe extractor, audio chunk + transcript + dịch EN, clip-features, `hybrid_search.py` gộp keyword + vector) — code tham khảo tốt cho phần index.
- **trungbac11/AIC-2024**: chia sẻ keyframes nén trên Kaggle — nguồn dataset tiện.
- ⚠️ Tránh nhầm: `NVIDIAAICITYCHALLENGE/...`, `AI City Challenge` là **cuộc thi khác** (CVPR, đô thị giao thông), không phải AI Challenge HCMC.

### 4.2. Các repo 2025–2026 (chi tiết đã có trong `docs/KISBranche-Features.md` và `docs/Schema-Reference.md`)

| Repo | Pipeline tóm tắt | Điểm khác biệt |
|---|---|---|
| **vkhanh2802/KISBranche** (2026) | Gemini query structuring → 3 luồng (SigLIP+Milvus / BM25 trên caption BLIP / MiniLM+FAISS) → object boost (SSDLite) → rerank Qwen2.5-VL | Có chế độ CPU-only (`KIS_DISABLE_VLM`), model pinning + manifest SHA-256, ~13 test file |
| **risc97/aic26-backend** (2026) | FastAPI + SQLite một nguồn sự thật + FTS5 + turbovec; SigLIP2/C2LIP/DINOv3/GTE, OWLv2 | Pack vector_id int64 gọn; schema chuẩn nhất để tham khảo (đã chốt làm nền của mình) |
| **ph4mHoa/aic2026-video-retrieval** (2026) | Không vector DB — inner product trên numpy mmap; Pydantic contract + JSON Schema hash | **Bài học VFR** (t_ms từ timebase, cấm nhân fps); sampling sai làm recall cap ~20% → benchmark; index full dataset ~5.6h / $0.83 (RTX 3090 thuê) |
| **ThuyHaLE/SceneSeek** (2026) | Jina CLIP v2 + FAISS HNSW/FlatIP + BM25 (pyvi) + RRF, FastAPI + React | Chạy **CPU-only** end-to-end; config-driven; chuẩn hóa event_id tại nguồn |
| **pgtomvn/AIC2026** (2026) | SigLIP2 + Qdrant + Meilisearch, weighted RRF, FastAPI + Streamlit | KIS+QA+TRAKE đầy đủ |
| **ledinhminhquan/Core-Vision** (2026) | SigLIP-2/MetaCLIP2 + FAISS + BM25, cross-encoder (BLIP-2 ITM) + Qwen3-VL rerank, Streamlit | Ablation có kiểm soát; 970 tests |

### 4.3. Bảng so sánh tổng hợp (2024 vs 2026)

| Thành phần | Các đội 2024 | Các đội 2026 | Xu hướng |
|---|---|---|---|
| Keyframe | BTC trích sẵn / SBD tự trích (Pumpkin, evento) | Tự trích + benchmark kỹ (ph4mHoa) | Shot-based > uniform |
| Embedding | CLIP BTC (L14/B32/B16), SigLIP, DFN5B, bigG, BEiT3 | SigLIP-2, Jina CLIP v2, DINOv3, GTE | SigLIP-2 family là de-facto |
| Vector store | FAISS / Qdrant / Milvus | FAISS / Qdrant / Milvus / **numpy mmap (bỏ DB)** | File-based FAISS đủ dùng cho cỡ dataset này |
| Text search | Elasticsearch (3 đội), FAISS | BM25, SQLite FTS5, Meilisearch | SQLite FTS5 đang thắng vì không cần service |
| OCR | EasyOCR, PaddleOCR | PaddleOCR, VietOCR, Vintern-1B, Gemini OCR | PaddleOCR phổ biến nhất |
| ASR | Whisper, Wav2Vec2-VLSP | faster-whisper, PhoWhisper | Model tiếng Việt được ưu tiên |
| Fusion | Hợp nhất multi-index, temporal scoring | **RRF/weighted-RRF chuẩn hóa** | RRF |
| Rerank | Temporal/color/hash filter | VLM (Qwen2.5-VL, Qwen3-VL), cross-encoder, Gemini | Rerank là tầng riêng, tắt được |
| LLM | Hiếm (Gemini ở evento) | **Gemini cho query structuring gần như chuẩn chung** | LLM = tầng query processing, không phải retrieval |
| Backend/UI | Flask/FastAPI + Jinja2/React/Streamlit | FastAPI + React/HTMX/Streamlit | FastAPI thống trị |
| Độ bền | Backup ES, Redis cache | Manifest checksum, model pinning, contract hash | Test + manifest ngày càng nghiêm |
| CPU-only | — (nhiều đội có GPU) | SceneSeek, KISBranche, ph4mHoa đều chạy được CPU | Đúng ràng buộc của mình |

### 4.4. Kỹ thuật đặc trưng cần lấy theo thứ tự giá trị

1. **Hybrid 3 kênh + RRF** — nền tảng, không đội nào bỏ.
2. **Benchmark keyframe sampling bằng recall** — tăng recall cap từ ~20% lên >90% chỉ nhờ chọn lưới đúng (ph4mHoa).
3. **Query structuring bằng Gemini** có cache/timeout/fallback — chuẩn 2026, hiếm ở 2024.
4. **Temporal-aware scoring** (Pumpkin 2024) — đặc biệt hợp vòng chung kết KIS vì mô tả sự kiện thường gồm nhiều cảnh liên tiếp; xếp sau vì cần shot/segment metadata.
5. **Temporally decomposed query** (ArtemisSearch): tách mô tả nhiều ý thành chuỗi câu con tìm lần lượt.
6. **Multi-embedder ensemble** (NaiveNotNice) — tăng recall nhưng tăng chi phí index; cân nhắc ở M3.
7. **Binary quantization** (Pumpkin) — chỉ cần khi index quá chậm trên CPU; dataset này FAISS HNSW đủ nhanh.

---

## Phần 5 — Plan đề xuất cho đội mình (ràng buộc: CPU-only, Gemini async, 2 người)

> Quyết định chi tiết đã chốt trong `docs/Plan-KIS-Final-Round.md` và `docs/Schema-Reference.md`; phần này tóm tắt và gắn với tổng hợp ở trên.

### 5.1. Kiến trúc đã chọn

```
Offline (chạy 1 lần, thuê GPU nếu cần):
  Keyframes (sampling benchmark) → SigLIP-2 embedding → FAISS (IndexIDMap2, vector_id int64)
  OCR (PaddleOCR, text phẳng mỗi keyframe) → SQLite FTS5 kép (có dấu + bỏ dấu)
  ASR (faster-whisper) → SQLite, theo khoảng thời gian
  Tất cả vào 1 SQLite (schema risc97, xem docs/Schema-Reference.md) + manifest.json

Runtime (CPU, < 2s):
  Query (append dần trong giờ thi) → Gemini structuring (cache/2s/fallback nguyên văn)
  → 3 luồng song song (FAISS visual / FTS5 OCR / FTS5 ASR)
  → weighted RRF → (optional) Gemini rerank top-20 (async, cache, off-able)
  → grid kết quả + nguồn match, khay ghim ứng viên
UI: FastAPI + frontend nhẹ; timer + "điểm nếu nộp đúng bây giờ"; zoom + scrub cho Q&A
Nộp bài: DRES client + mock server để rehearsal
```

### 5.2. Bỏ có chủ ý so với các repo tham khảo

| Bỏ | Vì sao | Quay lại khi |
|---|---|---|
| Milvus/Qdrant/Elasticsearch | Ops burden không đáng cho dataset vài trăm–nghìn video (ph4mHoa, SceneSeek chứng minh FAISS+SQLite đủ) | Nếu chuyển nền tảng lớn hơn |
| VLM local (Qwen2.5-VL) | Không GPU | Nếu có GPU |
| BLIP caption | OCR+ASR đã phủ; caption chỉ tăng recall biên | M3 nếu eval cho thấy recall thiếu |
| Object detection boost | Tốn công index, ích lợi biên | M3 optional (risc97 dùng OWLv2) |
| Temporal scoring (Pumpkin) | Cần shot segmentation + segment metadata | M3 optional — hợp vòng KIS nếu còn thời gian |

### 5.3. Milestones (đầy đủ trong Plan-KIS-Final-Round.md)

- **M0** — Bộ xương + eval harness (≥50 query tự sinh): index 20 video end-to-end, search < 2s, số recall@20 baseline. *Không hoãn.*
- **M1** — Đủ 3 kênh + RRF + Gemini structuring + UI append/ghim/scrub + mock DRES. Nghiệm thu: recall@20 ≥ 0.70, drill 1 Textual KIS < 5 phút.
- **M2** — Benchmark sampling, tune trọng số RRF, Gemini rerank, rehearsal đúng đồng hồ. Nghiệm thu: recall@20 ≥ 0.85, Textual KIS trung vị ≤ 2.5 phút.
- **M3** — Degraded mode (mất Gemini), latency p95 < 2s, runbook giờ thi, **freeze code**. Optional: temporal scoring, OCR box cho Q&A.

### 5.4. Phân công 2 người

| | Người A | Người B |
|---|---|---|
| Chính | Indexing (keyframe/OCR/ASR/FAISS/SQLite), search + RRF, eval harness | UI (grid/scrub/zoom/khay ghim/timer), DRES client + mock, query UX |
| Phụ | Gemini integration + cache | Script rehearsal, thu thập query eval |
| Chung | Viết query eval, drill mỗi milestone, review code chéo | |

### 5.5. Việc tuần đầu

1. **A**: script keyframe sampling + SigLIP-2 embed 20 video + FAISS search CLI (tham khảo `lnkhang1305` cho layout BTC, `tb-tian/AIChallenge2024` cho pipeline offline sạch).
2. **B**: FastAPI skeleton + grid UI + mock DRES + format `eval/queries.jsonl`.
3. **Chung**: chốt schema artifact theo `docs/Schema-Reference.md` §5 (đổi sau rất đắt) — đã chốt, giữ nguyên.

---

## Phần 6 — Tài liệu tham khảo (repo)

**2024**: [Pumpkin](https://github.com/VinhAlth/Pumpkin_AIC_2024) · [NaiveNotNice](https://github.com/chisngooo/Multimodal-Video-Retrieval-Engine-with-Vision-and-Text) · [evento/AIO_TOP10](https://github.com/MinLee0210/evento) · [ArtemisSearch](https://github.com/LoylP/AIC2024) · [xlinh2301_backend](https://github.com/xlinh2301/aic_backend) · [lnkhang1305](https://github.com/lnkhang1305/AIC2024_Video_Retrieval) · [JakeClark-chan](https://github.com/JakeClark-chan/HCMC_AI_Challenge2024_Image-Retrieval-QnA) · [QuanHoangNgoc-Final](https://github.com/QuanHoangNgoc/Image-based-Viewer-Web-and-Video-Retrieval-AIC-2024-HCMC-Final-) · [tb-tian](https://github.com/tb-tian/AIChallenge2024) · [trungbac11](https://github.com/trungbac11/AIC-2024) · [Ngố Team](https://github.com/KangaOnGit/HCMC_AI_Challenge2024_Image-Retrieval-QnA)

**2026**: [KISBranche](https://github.com/vkhanh2802/KISBranche) · [risc97/aic26-backend](https://github.com/risc97/aic26-backend) · [ph4mHoa](https://github.com/ph4mHoa/aic2026-video-retrieval) · [SceneSeek](https://github.com/ThuyHaLE/SceneSeek) · [pgtomvn](https://github.com/pgtomvn/AIC2026) · [Core-Vision](https://github.com/ledinhminhquan/Core-Vision-AIC2026)
