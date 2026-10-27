# Tham khảo Schema dữ liệu — các repo AI Challenge HCMC 2026

> Mục đích: chốt schema artifact (video_id / keyframe identity / thời gian / index & metadata) cho hệ thống mới.
> Nguồn: đọc trực tiếp mã nguồn 4 repo (KISBranche, risc97/aic26-backend, ph4mHoa/aic2026-video-retrieval, ThuyHaLE/SceneSeek). Ngày: 2026-10-25.

## 1. KISBranche — CSV mapping + JSON + .npy

**Định danh video**: regex `^L\d+_V\d+$`, ví dụ `L21_V001`. Group = phần trước `_` (`L21`).

**Mapping keyframe** (CSV mỗi video, `mapping/<group>/<video_id>.csv`):
```
n, pts_time, fps, frame_idx      # n = số thứ tự keyframe trong video (0-based)
                                  # frame_idx = số frame tuyệt đối trong video
                                  # pts_time = thời điểm (giây) lấy từ container
```

**Embedding**: `.npy` mỗi video, shape `(N, 768)` — N = số keyframe, thứ tự row khớp `n`. Layout: `clip-features-32/<group>/<video_id>.npy`.

**Caption record** (JSON): `{video_id, keyframe_n, frame_id, image_path, caption, details, ocr_text}` — có validate chéo `frame_id` với mapping CSV, phát hiện duplicate/gap. Text retrieval = ghép `caption + details + "[OCR]: " + ocr_text`.

**Object JSON**: 5 mảng song song kiểu Open Images — `detection_scores`, `detection_class_names` (MID `/m/xxx`), `detection_class_entities`, `detection_boxes` (chuỗi `ymin,xmin,ymax,xmax` normalized), `detection_class_labels`.

**Điểm đáng học**: resolve artifact có phát hiện trùng lặp (nhiều vị trí tìm thấy cùng 1 file → raise); manifest SHA-256 cho caption artifacts.

## 2. risc97/aic26-backend — SQLite làm nguồn sự thật duy nhất

**Đầy đủ nhất để tham khảo. Schema:**
```sql
videos     (video_id PK, video_path, duration_ms INT, fps REAL)
segments   (segment_id PK, video_id FK, frame_start INT, frame_end INT)
keyframes  (keyframe_id TEXT, video_id FK, frame_idx INT, timestamp_ms INT,
            image_path TEXT, segment_id FK NULL,
            PRIMARY KEY (video_id, keyframe_id), UNIQUE (video_id, frame_idx))
transcripts(transcript_pk PK, transcript_id, video_id FK,
            time_start_ms INT, time_end_ms INT, text)
ocr        (ocr_id PK, video_id, keyframe_id, text, UNIQUE (video_id, keyframe_id))

-- FTS5 + trigger đồng bộ tự động cho transcripts và ocr:
CREATE VIRTUAL TABLE transcripts_fts USING fts5(
    text, content='transcripts', content_rowid='transcript_pk',
    tokenize="unicode61 remove_diacritics 0");
```

**Đóng gói vector ID thành int64** (không cần sidecar metadata cho FAISS):
```
vector_id = video_head × 10_000 + keyframe_n
video_head = pack(prefix_letter, group, num)   # L21_V001 → 21001
# 21001_0001 → 210010001 ; giải ngược được 2 chiều, regex video_id: ^([LSNM])(\d+)[-_]V(\d+)$
```
Dataset thật có nhiều tiền tố batch: `L`, `S`, `N`, `M` (ví dụ `S01-V001`, `N001-V001`).

**Điểm đáng học**: một SQLite chứa hết metadata + FTS full-text; keyframe có cả `frame_idx` lẫn `timestamp_ms`; UNIQUE(video_id, frame_idx) chống trùng.

## 3. ph4mHoa/aic2026-video-retrieval — Pydantic contract, thời gian là danh tính chính

**KeyframeRef** (đối tượng danh tính keyframe dùng xuyên suốt hệ thống):
```json
{ "video_id": "L21_V001",     // pattern ^L\d{2}_V\d{3}$  (bắt buộc)
  "t_ms": 1234567,            // integer ms  (BẮT BUỘC — trường danh tính chính)
  "kf_n": 12,                 // tùy chọn
  "frame_idx": 3456 }         // tùy chọn
```

**Interval**: `{t_start_ms, t_end_ms}` (integer ms).

**DatasetManifest** mỗi pack: `pack_id` (`^L\d{2}$`), `n_videos`, `n_keyframes`, `sha256_by_file` (checksum từng file) + **VideoProbe** `{video_id, fps_source, vfr, duration_ms}`.

**Bài học lớn nhất của họ**: video trong dataset là **VFR (variable frame rate)** — không được nhân `frame_idx / fps` ra thời gian; phải tra bảng `timebase/` mỗi video. Đây là lý do `t_ms` là trường bắt buộc còn `frame_idx` chỉ là tùy chọn.

Toàn bộ API dùng schema JSON có version (`contract/VERSION`) + hash verification — frontend và backend cùng kiểm một contract.

## 4. SceneSeek — FAISS + sidecar dict, config-driven

**Mô hình**: FAISS index (`hnsw_jinaclipv2`, `flatip_dangvantuan`...) + **sidecar `info_dict`** (dict Python row → metadata frame). Khi load: `assert index.ntotal == len(info_dict)` — fail ngay nếu index và metadata lệch.

**Config-driven** (`config/databases.json`, `config/models.json`): mỗi index một key (đường dẫn, index_type, `hnsw_ef_search`...); đổi model/dataset chỉ sửa JSON. BM25 lưu object `BM25Okapi` pickle; FAISS FlatIP nhỏ thì build lại từ `embeddings.pt` lúc khởi động (vài giây) thay vì giữ file binary.

**Chuẩn hoá event_id tại nguồn**: transcript thô có event_id có gap và không local per video → sort theo `(video_id, start)`, gán lại id local 0-based gap-free, giữ `source_event_id` để debug. Bài học: **chuẩn hoá id một lần ở chỗ load dữ liệu**, mọi module phía sau yên tâm giả định id liên tục.

## 5. Các điểm chung & quyết định schema cho hệ thống mới (ĐÃ CHỐT)

**Quyết định (2026-10-25, sau buổi grill-me): lấy schema risc97/aic26-backend làm nền**, với 3 điều chỉnh:

1. **Bỏ bảng `segments`** — plan không có shot detection; cần thì thêm sau bằng migration rẻ (chỉ metadata).
2. **OCR text phẳng như risc97** — mỗi keyframe 1 chuỗi, không lưu boxes. Đánh đổi: không làm được "zoom tới vùng chữ" cho Q&A (đã chọn human-in-the-loop đọc số nên chấp nhận). Nếu đổi ý sau khi đã chạy OCR toàn dataset thì phải chạy lại OCR.
3. **FTS5 kép 2 cột** — 1 cột token hóa giữ dấu (`unicode61 remove_diacritics 0`) + 1 cột bỏ dấu (`remove_diacritics 2`), search union cả hai. Lý do: query khi thi có thể gõ không dấu.

**2 điều kiện bắt buộc giữ nguyên tinh thần của các repo tham khảo:**
- `t_ms` lấy từ `pts_time` của mapping CSV/ffprobe, **không bao giờ** tính từ `fps` (bài học VFR của ph4mHoa). Cột `fps` trong bảng `videos` chỉ để tham khảo.
- Vector ID đóng gói int64 kiểu risc97 phải **kiểm tra biên** lúc ingest: ≤10.000 keyframe/video, ≤1.000 video/pack, đủ các tiền tố `L/S/N/M`; thêm `assert FAISS.ntotal == COUNT(keyframes)` lúc khởi động + `manifest.json` (model revision, checksum).

**Schema SQLite cuối cùng của hệ thống mới:**
```sql
videos     (video_id TEXT PK, pack_id TEXT, video_path TEXT,
            duration_ms INT, fps_source REAL, vfr INT)
keyframes  (video_id TEXT, kf_n INT, frame_idx INT, t_ms INT NOT NULL, image_path TEXT,
            PRIMARY KEY (video_id, kf_n), UNIQUE (video_id, frame_idx))
ocr        (video_id TEXT, kf_n INT, text TEXT,          -- text phẳng, không boxes
            PRIMARY KEY (video_id, kf_n))
transcripts(video_id TEXT, seg_id INT, t_start_ms INT, t_end_ms INT, text TEXT,
            PRIMARY KEY (video_id, seg_id))

-- FTS5 kép cho cả ocr và transcripts (mẫu cho transcripts; ocr làm tương tự):
CREATE VIRTUAL TABLE transcripts_fts USING fts5(
    text, text_nodia,                          -- bản có dấu + bản bỏ dấu
    content='transcripts', content_rowid='rowid',
    tokenize="unicode61 remove_diacritics 0");
-- cột text_nodia dùng tokenizer remove_diacritics 2 (index riêng hoặc bảng FTS thứ 2)
-- + trigger đồng bộ insert/update/delete như risc97

-- FAISS: IndexIDMap2, vector_id = pack(video_id, kf_n) kiểu risc97
-- (video_head × 10_000 + kf_n, video_head pack prefix L/S/N/M + group + num)
```

| Quyết định | Các repo làm gì | Chọn cho mình |
|---|---|---|
| Định danh video | Giữ mã BTC `L##_V###` (+ S/N/M), regex validate | Giữ nguyên, regex `^([LSNM]\d+)[-_]V\d+$` |
| Danh tính keyframe | ordinal `kf_n` (KISBranche), `frame_idx` (risc97), `t_ms` bắt buộc (ph4mHoa) | Lưu đủ 3: PK `(video_id, kf_n)`, `frame_idx` UNIQUE, `t_ms NOT NULL` |
| Đơn vị thời gian | Integer ms mọi repo | Integer ms |
| Frame → time | pts_time CSV (KISBranche), timebase/ vì VFR (ph4mHoa) | Ingest từ `pts_time`/ffprobe, cấm nhân fps |
| Metadata store | SQLite+FTS5 (risc97) / JSON+npy (KISBranche) / pickle (SceneSeek) | SQLite một nguồn sự thật (risc97) |
| FAISS ↔ metadata | Pack int64 (risc97) / sidecar+assert (SceneSeek) | IndexIDMap2 + pack int64 + assert ntotal == COUNT |
| Toàn vẹn | SHA-256 manifest (ph4mHoa, KISBranche) | `manifest.json` mỗi lần index |
| OCR | FTS5 theo keyframe (risc97) | Text phẳng + FTS kép (không boxes) |
| Từ khóa tiếng Việt | unicode61 (risc97), pyvi+BM25 (SceneSeek) | FTS5 kép có dấu + bỏ dấu |
| Layout file | `<pack>/<video_id><suffix>` | Giữ layout nhóm của BTC |
