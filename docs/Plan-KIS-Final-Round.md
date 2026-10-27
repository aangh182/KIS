# Plan: Hệ thống Retrieval — Vòng chung kết AI Challenge HCMC 2026

> Tài liệu quyết định sau buổi phỏng vấn grill-me (2026-10-25).
- **Phạm vi**: Textual KIS (5 phút/query, mô tả cung cấp dần), Video KIS (4 phút/query, mô tả lại bằng tiếng Việt), Q&A (5 phút/query, đáp án là chuỗi số 0–9). Nộp bài qua DRES client.
- **Điểm chấm**: `Score = (50 − 10×ws) + 50×(1 − t/tL)` — mỗi lần nộp sai mất 10 điểm chính xác, điểm thời gian giảm tuyến tính. → Hệ thống phải **nhanh** và phải **giúp quyết định khi nào nộp**.

## 1. Ràng buộc đã chốt (từ buổi phỏng vấn)

| Ràng buộc | Giá trị | Hệ quả kiến trúc |
|---|---|---|
| Codebase | Xây mới, tham khảo KISBranche | Không mang nợ kỹ thuật; chọn stack tối giản cho CPU |
| Phần cứng giờ thi | Server CPU vừa phải, không GPU | Mọi embedding phải **precompute offline**; runtime chỉ encode text query; không VLM local |
| Mạng | Ổn định, gọi được Gemini | Gemini cho query structuring + rerank (async, có cache + fallback) |
| Nhân sự | 2 người, cả 2 viết code | Phân công 2 trục: backend/indexing vs UI/DRES |
| Workflow thi | 1 phiên chung, 2 người cùng xem | UI đơn luồng, keyboard-first, không cần multi-session sync |
| Dữ liệu | Có full dataset | Index ngay ở M0/M1 |
| Eval | Chưa có query + GT | Phải xây eval harness từ đầu (M0, không được hoãn) |
| Thời gian | Không có lịch cụ thể | Kế hoạch theo milestone M0–M3, mỗi milestone có tiêu chí nghiệm thu |

## 2. Kiến trúc hệ thống

### 2.1. Nguyên tắc設 kế
1. **Offline nặng, runtime nhẹ**: mọi thứ tốn GPU (embedding keyframe, OCR, ASR) chạy một lần offline (thuê GPU nếu cần — tham khảo ph4mHoa: index full dataset ~5.6h với $0.83 trên RTX 3090). Runtime trên server CPU chỉ làm: encode text query, tìm FAISS/BM25, fuse, render.
2. **Ít moving parts**: không Milvus/Qdrant (ops burden cao cho dataset cỡ vài trăm–vài nghìn video) — dùng **FAISS (file) + SQLite** như SceneSeek/ph4mHoa. Không Docker Compose phức tạp trong giờ thi.
3. **Mọi tính năng có fallback**: Gemini timeout 2s → bỏ qua silently, kết quả vẫn trả từ pipeline local.
4. **UI tối ưu cho đồng hồ**: kết quả trong <2 giây, phím tắt, timer + EV nộp bài hiển thị luôn màn hình.

### 2.2. Pipeline retrieval (runtime, CPU)

```
Query tiếng Việt (gõ dần theo thời gian thi)
  │
  ├─ Gemini Query Structuring (cache theo text, timeout 2s, fallback: dùng nguyên văn)
  │    → tách thuộc tính (địa điểm, nhân vật, vật thể, chữ trên màn hình, thời gian)
  │    → bản dịch EN cho CLIP text tower, các keyword cho BM25
  │
  ├─ Luồng 1 — Visual: SigLIP/SigLIP-2 text encode (CPU, ~100ms) → FAISS HNSW top-k
  ├─ Luồng 2 — OCR text: BM25/FTS5 trên OCR (PaddleOCR, index offline)
  ├─ Luồng 3 — ASR text: BM25/FTS5 trên transcript (faster-whisper, index offline)
  │
  ├─ Fusion: weighted RRF (trọng số tune ở M2 trên eval set)
  ├─ Gemini Rerank top-20 (async, optional, có cache — bỏ qua nếu lỗi)
  │
  └─ Kết quả: grid keyframe + điểm hỗn hợp + nguồn match (visual/OCR/ASR — để operator biết tin cân nào)
```

**Cấu trúc index (offline, chạy một lần):**
- `keyframes/`: sampling benchmark ở M2 (1 fps vs lưới 250ms vs shot-based) — quyết định bằng số đo recall, không bằng cảm tính.
- `embeddings.faiss`: image embeddings SigLIP-2 (hoặc organizer CLIP nếu có sẵn).
- `text.db` (SQLite FTS5): bảng OCR theo keyframe + transcript theo đoạn thời gian, map về keyframe.
- `manifest.json`: checksum + model revision (học từ KISBranche) để tái tạo index deterministic.

**Bỏ có chủ ý so với KISBranche / các repo tham khảo:** Milvus (thay FAISS), Qwen2.5-VL local rerank (không GPU), BLIP caption (OCR+ASR đã phủ; cân nhắc thêm ở M3 nếu recall thiếu), object detection boost (M3 optional nếu còn thời gian — risc97 dùng OWLv2).

### 2.3. Xử lý từng dạng query

| Dạng | Luồng người | Luồng máy |
|---|---|---|
| **Textual KIS** | Mô tả hiện dần → operator **append** thêm text vào ô query, search lại (không gõ lại từ đầu). Ghim kết quả nghi ngờ vào "khay ứng viên" | Query structuring + 3 luồng + rerank; nút "chỉ tìm trong khay ứng viên" khi mô tả mới về |
| **Video KIS** | Xem 20s → gõ mô tả tiếng Việt nhiều chi tiết, **ưu tiên chữ xuất hiện trên màn hình (banner, biển hiệu) vì OCR là kênh mạnh nhất** | Cùng pipeline text; UI có sẵn checklist thuộc tính (bối cảnh/ trang phục/ chữ/ hành động) để gõ nhanh |
| **Q&A** | Tìm đúng frame như KIS → **zoom + scrub frame kề bên để tự đọc chuỗi số** → gõ vào DRES | Không auto-trả lời (quyết định human-in-the-loop); M3 thêm gợi ý OCR vùng có số (PaddleOCR box) nếu kịp |

### 2.4. UI (1 phiên chung, keyboard-first)

- **Màn hình chính**: hộp query + grid kết quả (thumbnail lớn đủ đọc chữ trên banner) + sidebar khay ứng viên đã ghim.
- **Xem chi tiết 1 keyframe**: frame lớn + **cầu trượt frame kề trước/sau** (xem sự kiện chuyển động — quan trọng để xác nhận trước khi nộp) + zoom cho Q&A.
- **Panel DRES + đồng hồ**: đếm ngược tL, số lần nộp sai ws, và **số điểm nhận được nếu nộp đúng ngay lúc này** `(50−10ws) + 50(1−t/tL)` — hiện lớn, cập nhật mỗi giây.
- **Hỗ trợ quyết định nộp** (quan trọng ngang tìm kiếm): checklist xác minh (OCR khớp chữ? khung cảnh khớp? frame kề hợp lý?) + gợi ý EV: nộp sai chỉ mất 10 điểm → nộp sớm khi tin cậy ≥~60% thường có kỳ vọng dương. Nguyên tắc luyện ở M2: **nộp nhanh đúng 1 lần > chờ tuyệt đối**.
- Stack: FastAPI + HTMX/React nhẹ (1 người làm riêng), không cần WebSocket.

### 2.5. DRES client
- Module riêng gọi HTTP API DRES (`dres-dev/DRES`): submit (video id + frame), nhận kết quả đúng/sai.
- Mock DRES server (FastAPI giả) để **rehearsal không cần server thật** — dùng cho mọi buổi luyện từ M1.
- Hiển thị phản hồi sai ngay UI (đội biết mình đã mất 10 điểm và tiếp tục tìm).

## 3. Milestones (tiêu chí nghiệm thu đo được)

### M0 — Bộ xương + Eval harness (quan trọng nhất, làm trước tiên)
- [ ] Layout dữ liệu + script index 20 video đầu tiên end-to-end: keyframe → SigLIP → FAISS; search text → grid kết quả (UI thô).
- [ ] **Eval harness**: script sinh bộ query luyện tập — chọn ngẫu nhiên keyframe làm GT, 2 người viết mô tả theo đúng phong cách thể lệ (nhiều ý, cung cấp dần thành từng "reveal"), lưu `eval/queries.jsonl`. ≥50 query ở M0, tăng lên ≥150 ở M2 (cả 3 dạng).
- [ ] Metric: recall@20, MRR, latency p95. Báo cáo tự động sau mỗi lần chạy.
- **Nghiệm thu**: search 1 query < 2s trên server CPU; có số recall@20 đầu tiên (baseline).
- *Phân công: A làm indexing + search; B làm UI thô + eval harness format.*

### M1 — Đủ 3 kênh + fusion + DRES mock
- [ ] OCR offline (PaddleOCR) + ASR offline (faster-whisper) → FTS5; gộp BM25 + visual bằng RRF.
- [ ] Gemini query structuring (cache + timeout + fallback nguyên văn).
- [ ] UI: append query, khay ghim, xem chi tiết + scrub, panel DRES mock + timer + điểm-if-correct-now.
- **Nghiệm thu**: recall@20 trên eval ≥ **0.70**; chạy 10 query liên tục không crash; drill đầu tiên: giải 1 Textual KIS < 5 phút.

### M2 — Chất lượng + chiến thuật
- [ ] Benchmark sampling keyframe (1fps / 250ms / shot) → chọn bằng recall, re-index full dataset.
- [ ] Tune trọng số RRF + thêm/bớt channel bằng ablation trên eval (học cách Core-Vision/ph4mHoa làm ablation có kiểm soát).
- [ ] Gemini rerank top-20 (async, cache, off-able) — chỉ giữ nếu recall/precision tăng có ý nghĩa.
- [ ] Rehearsal định kỳ (2 người, mock DRES, đúng đồng hồ thể lệ): đo **thời gian giải trung bình, số lần nộp sai, điểm mô phỏng**.
- **Nghiệm thu**: recall@20 ≥ **0.85**; drill: Textual KIS trung vị ≤ 2.5 phút, Video KIS ≤ 3 phút; ≤1 lần nộp sai trung bình.

### M3 — Cứng hóa + đóng băng
- [ ] Degraded mode: mất Gemini → vẫn search được (đã có fallback), diễn tập cutoff mạng.
- [ ] Latency budget: p95 < 2s từ gõ đến render; warm-up index khi khởi động.
- [ ] Runbook giờ thi: checklist mở hệ thống, phím tắt, vai từng người, quy tắc nộp (khi nào nộp, ai bấm).
- [ ] **Freeze code**: không thêm model/tính năng mới sau điểm này; chỉ sửa bug.
- *Optional chỉ nếu dư thời gian*: OCR box hỗ trợ đọc số cho Q&A; object metadata boost.

## 4. Phân công 2 người

| | Người A | Người B |
|---|---|---|
| Chính | Indexing pipeline, search + fusion, eval harness | UI (grid/chi tiết/scrub/zoom), DRES client + mock, query UX |
| Phụ | Gemini integration + cache | Script rehearsal, thu thập query eval |
| Chung | Viết query eval, chạy drill mỗi milestone, review code nhau | |

## 5. Rủi ro & đối sách

| Rủi ro | Đối sách |
|---|---|
| Gemini lag/rate-limit giữa giờ thi | Cache mọi query, timeout 2s, fallback nguyên văn; mọi kênh local vẫn chạy độc lập |
| Server CPU yếu, encode text chậm | Benchmark ngay M0; nếu >500ms thì quantize/đổi encoder text nhỏ hơn (mặt bằng: SceneSeek chạy CPU-only với Jina CLIP v2) |
| Video KIS mô tả thiếu chi tiết → recall thấp | Drill viết mô tả theo checklist thuộc tính; nhấn mạnh chữ trên màn hình (kênh OCR); chấp nhận search nhiều lần trong 4 phút |
| Đọc sai số ở Q&A | Zoom + scrub + xác nhận chéo 2 người trước khi nộp; mỗi lần sai −10 điểm |
| Over-engineering chạy quá giờ | Mỗi milestone có nghiệm thu đo được; freeze ở M3; "optional" chỉ làm khi xong sớm |
| Bias tự đánh giá (tự viết query, tự chấm) | Query viết cách nhau ≥1 tuần sau khi xem video / mô tả theo template thể lệ; đo recall tự động, không cảm tính |

## 6. Tech stack tóm tắt (theo tổng hợp docs/KISBranche-Features.md)

| Thành phần | Lựa chọn | Vì sao (tham chiếu các đội 2026) |
|---|---|---|
| Embedding thị giác | SigLIP-2 (precompute offline) | Chuẩn de-facto của mọi đội |
| Vector search | FAISS HNSW (file) | SceneSeek, Core-Vision cùng chọn; không cần ops vector DB |
| OCR / ASR | PaddleOCR + faster-whisper (offline) | Cặp phổ biến nhất (pgtomvn, ph4mHoa) |
| Từ khóa | SQLite FTS5 | Nhẹ, không service riêng (ph4mHoa) |
| Fusion | weighted RRF | Phương pháp chung của tất cả các đội |
| Rerank | Gemini top-20 (optional, async) | Thay VLM local vì không GPU |
| Query processing | Gemini structuring + cache | Chuẩn chung (KISBranche, Core-Vision, pgtomvn) |
| Backend / UI | FastAPI + frontend nhẹ, 1 session | Thống nhất hệ sinh thái cuộc thi |
| Nộp bài | DRES client + mock server | Thể lệ vòng chung kết |
| Kiểm thử | pytest/eval harness + rehearsal drills | ph4mHoa (538 test), Core-Vision (970 test) |

## 7. Việc bắt đầu ngay (tuần đầu)
1. Người A: script sampling keyframe + SigLIP embed 20 video + FAISS search CLI.
2. Người B: FastAPI skeleton + grid UI + mock DRES client + format `eval/queries.jsonl`.
3. Chung: thống nhất schema artifact (keyframe id, video id, timestamp) — đổi sau đắt.
