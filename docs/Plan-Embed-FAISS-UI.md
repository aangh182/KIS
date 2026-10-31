# Plan: Embed SigLIP-2 → FAISS → Search CLI → Web UI

> Phạm vi: M0-B (một phần) — thêm luồng thị giác để test retrieval trên 20 video đã index.
> OCR / ASR / FTS / RRF / Gemini / DRES / timer **để sau**.
> Tài liệu nền: `docs/Plan-Self-Keyframe.md` §5–§9, `docs/Plan-KIS-Final-Round.md` §2, `docs/Schema-Reference.md` §5.

## 0. Trạng thái triển khai (2026-10-31)

- **Đã xong code**: `kis embed`, `kis index build`, `kis search`, `kis serve` (FastAPI + UI 1 trang);
  unit test **27/27 pass**.
- **Đã chạy thật trên partial index**: 2/20 video, **1.428 vector 768-d fp16**;
  FAISS `IndexIDMap2(FlatIP)` 4,4 MB; `/api/video` trả **206 Range** đúng.
- **Đang chạy nền**: embed 19 video còn lại. Throughput đo được **~1,4–1,5 ảnh/s**
  trên Xeon E5-2690 (12 luồng, không AVX2); int8 dynamic crash vì thiếu AVX2.
- **Search (CPU rảnh)**: encode ~65–75 ms + FAISS ~2–20 ms → tổng <100 ms.
- **Chất lượng visual**: caption ngắn tiếng Anh tốt (vali/logo/MC áo trắng);
  tiếng Việt thô yếu. Prompt template `This is a photo of {}.` bật mặc định;
  dịch VI→EN (MarianMT) có code nhưng **mặc định tắt** vì chưa cải thiện top-1 rõ rệt.
  Mean-centering không giúp giảm hubness trên mẫu nhỏ.
- **Còn lại**: chờ embed xong → `kis index build` strict (14.061 vector);
  `eval/queries.jsonl` + recall@20; OCR/ASR/FTS/RRF.

## 1. Mục tiêu & nghiệm thu

| Hạng mục | Nghiệm thu |
|---|---|
| `kis embed` | 20 video / **14.061 keyframe** → 20 shard `.npy` fp16, dim 768, vector chuẩn hoá L2 |
| `kis index build` | `index/faiss.index` có `ntotal == 14.061 == COUNT(keyframes)`, manifest đầy đủ |
| `kis search` | trả top-20, score giảm dần, in latency encode/search/total |
| `kis serve` + UI | search bằng tiếng Việt, grid thumbnail, click xem keyframe lớn, mở video đúng timestamp |
| Độ bền | thiếu/lệch shard → build fail rõ ràng; load index assert `ntotal` khớp manifest + DB |
| Test | unit test mới cho normalize/shard/index id-map/format kết quả; 15 test cũ vẫn pass |

Chạy trên **20 video hiện có** (`Dataset/batch2`, 14.061 keyframe). Full 363 video là bước sau.

## 2. Quyết định đã chốt

| Quyết định | Chọn | Lý do |
|---|---|---|
| Model | `google/siglip2-base-patch16-384` | Docs đã chốt; 768-d; đa ngữ (109 ngôn ngữ) nên query tiếng Việt không cần dịch; ảnh 384 khớp keyframe |
| Index | `faiss.IndexIDMap2(faiss.IndexFlatIP(768))`, id = `keyframe_pk` | Exact như FlatIP; id tường minh nên chịu được gap pk sau `--force`; search trả thẳng pk để join SQLite |
| Chuẩn hoá | L2-normalize cả ảnh và text, lưu fp16 | Inner product = cosine; tiết kiệm đĩa |
| UI | FastAPI + 1 file HTML/JS tĩnh | Search + grid + detail + video; không DRES/timer/khay ghim |
| Phạm vi dữ liệu | 20 video / 14.061 kf | Vòng lặp nhanh; lệnh resumable để chạy full sau |

## 3. Điểm mâu thuẫn với docs (phải ghi rõ)

1. **`docs/Plan-Self-Keyframe.md` §4 nói `IndexFlatIP` + “vị trí = keyframe_pk”**. Điều này chỉ đúng nếu pk liên tục `1..N`. Thực tế `db.replace_keyframes()` `DELETE` rồi `INSERT` lại khi `--force`/reindex một phần → pk có lỗ hổng, vị trí ≠ pk. **Chọn `IndexIDMap2` bọc `IndexFlatIP`**: vẫn exact, chỉ thêm map id.
2. **Docs nói `.npy` shard theo pack**. Chọn **shard theo video** (`index/embeddings/<pack>/<video_id>.npy`) để resume/rebuild đúng video, khớp layout keyframe hiện có.
3. **Docs §7 mô tả UI M1 đầy đủ** (pin tray, timer, EV, DRES). Lần này chỉ làm **UI M0.5**: search + grid + detail + video link; các phần kia để M1.
4. **Docs §8 yêu cầu `eval/queries.jsonl` ≥50 query + recall@20**. Lần này chưa làm harness recall; chỉ đo latency end-to-end. Ghi vào “việc tiếp theo”.

## 4. Artifact & schema

Không đổi schema SQLite (dùng `keyframes.keyframe_pk` sẵn có).

```
index/
  embeddings/<pack>/<video_id>.npy      fp16, shape (N, 768), row order = kf_n tăng dần
  embeddings/<pack>/<video_id>.json     sidecar: model, dim, count, keyframe_pks[], sha256
  faiss.index                           IndexIDMap2(IndexFlatIP), id = keyframe_pk
  manifest.json                         model, dim, counts, checksums, sampling, partial
```

Sidecar JSON (ví dụ):
```json
{"video_id":"L13_V001","model":"google/siglip2-base-patch16-384","dim":768,
 "dtype":"float16","count":689,"keyframe_pks":[1,2,...],"sha256":"...","created_at":"..."}
```

`manifest.json`:
```json
{"built_at":"...","git_sha":"...","partial":false,
 "embed":{"model":"google/siglip2-base-patch16-384","dim":768,"dtype":"float16",
          "normalized":true,"transformers":"..."},
 "index":{"type":"IndexIDMap2(IndexFlatIP)","ntotal":14061,"faiss_sha256":"..."},
 "counts":{"videos_indexed":20,"keyframes_indexed":14061,"keyframes_db":14061},
 "sampling":{"profile":"hybrid","grid_s":3,"safety_grid_s":2},
 "checksums":{"meta_db_sha256":"..."}}
```

## 5. Module & CLI

### File mới

```
src/kis/ingest/embed.py     load model, embed ảnh theo video, ghi .npy + sidecar
src/kis/index/__init__.py
src/kis/index/build.py      đọc shard → FAISS + manifest + assert tính nhất quán
src/kis/index/store.py      FaissStore: load/validate/search id→pk
src/kis/search/__init__.py
src/kis/search/engine.py    encode text + FAISS + join SQLite → kết quả
src/kis/web/__init__.py
src/kis/web/app.py          FastAPI app + API + static mounts
src/kis/web/static/index.html
tests/test_embedding.py
tests/test_index_store.py
tests/test_search_engine.py
docs/Plan-Embed-FAISS-UI.md  (bản plan này, tiếng Việt, đặt trong repo)
```

### File sửa

| File | Thay đổi |
|---|---|
| `pyproject.toml` | `embed`: thêm `transformers>=4.50`; `web = ["fastapi>=0.115","uvicorn>=0.30"]`; thêm pytorch CPU index + `[tool.uv.sources]` |
| `src/kis/config.py` | `embed_model`, `embed_batch`, `embed_dtype`, `faiss_path`, `manifest_path`, `embeddings_dir`, `web_host`, `web_port`, `search_topk` |
| `src/kis/db.py` | helper `get_keyframes_for_video()`, `count_keyframes()`, `get_keyframes_by_pks()`; `assert_index_counts()` |
| `src/kis/cli.py` | thêm `embed`, `index build`, `search`, `serve` |
| `README.md` | cập nhật trạng thái, cấu trúc, cách chạy, kết quả đo |

### CLI

```bash
kis embed [--batch 32] [--limit N] [--video-id ID] [--force] [--model ID]
kis index build [--allow-partial] [--force]
kis search "mô tả tiếng Việt" [--k 20] [--json]
kis serve [--host 127.0.0.1] [--port 8000]
```

## 6. Chi tiết triển khai

### 6.1. `ingest/embed.py`

- Load `AutoModel.from_pretrained(model_id, attn_implementation="sdpa").eval()` + `AutoProcessor.from_pretrained(model_id)`, CPU.
- `torch.set_num_threads(max(1, os.cpu_count()))`; `torch.inference_mode()`.
- Với mỗi video: `db.get_keyframes_for_video()` ORDER BY `kf_n`; mở `image_path` (đã relative với `index/`), PIL RGB.
- `processor(images=batch, return_tensors="pt")` → `get_image_features` → `F.normalize(p=2, dim=-1)` → `np.float16`.
- Ghi `.npy` + sidecar JSON (kèm `keyframe_pks`, `sha256`).
- Resume: bỏ qua shard nếu JSON tồn tại, `model` khớp, `count` khớp, `sha256` khớp, và `keyframe_pks` khớp DB (trừ `--force`).
- Log: số ảnh, elapsed, img/s, skipped.

Lưu ý SigLIP-2: processor tự lowercase + pad/truncate `max_length=64`; **không override** các default này khi encode text.

### 6.2. `index/build.py`

- Quét `index/embeddings/*/*.json`; load `.npy`, cast `float32`.
- Validate: `dim == 768`, `count == len(keyframe_pks)`, mọi pk tồn tại trong DB, không trùng pk.
- Gom tất cả vector + id; tạo `IndexIDMap2(IndexFlatIP(dim))`, `add_with_ids`.
- Strict (mặc định): yêu cầu `set(ids) == set(keyframe_pk trong DB)` và `ntotal == COUNT(keyframes)`.
- `--allow-partial`: chỉ build trên shard có sẵn; manifest ghi `partial: true` + danh sách video.
- Ghi `faiss.index` + `manifest.json` (sha256 của faiss + meta.db).
- Fail sớm với thông điệp rõ nếu thiếu shard / lệch count.

### 6.3. `index/store.py`

- `FaissStore.load()`: đọc manifest + `faiss.read_index`; assert:
  - `index.ntotal == manifest.index.ntotal`
  - nếu `partial == false`: `index.ntotal == db.count_rows("keyframes")`
  - `manifest.embed.model == settings.embed_model`
- `search_ids(query_vector, k) -> list[(pk, score)]`.

### 6.4. `search/engine.py`

- `SearchEngine` giữ model + processor + `FaissStore` (load 1 lần khi app/CLI khởi động).
- `search(query, k)`:
  1. `processor(text=[query], return_tensors="pt")` → `get_text_features` → L2 normalize.
  2. FAISS search top-k → `(pk, score)`.
  3. `db.get_keyframes_by_pks(pks)` join `videos` + `shots`; giữ nguyên thứ tự FAISS.
  4. Build dict kết quả: `keyframe_pk, video_id, kf_n, t_ms, t_label, score, image_url, thumb_url, shot, video_url`.
- Đo `encode_ms`, `search_ms`, `total_ms`.
- Nếu `k > ntotal` thì clamp.

### 6.5. `web/app.py` + `static/index.html`

- `create_app(settings)`; load engine 1 lần (lifespan), fail fast nếu index chưa build.
- Mount:
  - `/media/keyframes` → `index/keyframes`
  - `/media/thumbnails` → `index/thumbnails`
- API:
  - `GET /api/health` → `{videos, keyframes, ntotal, model, dim, partial}`
  - `GET /api/search?q=...&k=20` → `{query, latency_ms, results[]}`
  - `GET /api/keyframe/{pk}` → metadata 1 keyframe
  - `GET /api/video/{video_id}` → `FileResponse(video_path)` (hỗ trợ Range để seek)
- UI 1 file:
  - Ô query + select k; `Enter` để search; hiện latency.
  - Grid thumbnail; mỗi card: video_id, `mm:ss.mmm`, score.
  - Click card → panel lớn hiện keyframe 384px phóng to + metadata.
  - Link “mở video” → `/api/video/<id>#t=<seconds>`.
  - Phím: `/` focus, `←/→` chọn card, `Enter` mở, `Esc` đóng.
- Chỉ bind `127.0.0.1` mặc định.

### 6.6. `config.py`

```python
embed_model: str = "google/siglip2-base-patch16-384"
embed_batch: int = 32
embed_dtype: str = "float16"
search_topk: int = 20
web_host: str = "127.0.0.1"
web_port: int = 8000

@property
def embeddings_dir(self) -> Path: return self.index_dir / "embeddings"
@property
def faiss_path(self) -> Path: return self.index_dir / "faiss.index"
@property
def manifest_path(self) -> Path: return self.index_dir / "manifest.json"
```

### 6.7. `pyproject.toml`

```toml
[project.optional-dependencies]
embed = ["faiss-cpu>=1.9", "torch>=2.4", "transformers>=4.50"]
web   = ["fastapi>=0.115", "uvicorn>=0.30"]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Mục đích: tránh tải torch CUDA ~2,5 GB; chỉ lấy CPU wheel. `numpy` hiện là 2.5.4 — kiểm tra `import faiss` sau sync, nếu lệch ABI thì nâng `faiss-cpu>=1.11`.

## 7. Thứ tự thực hiện

1. **Deps + config**: sửa `pyproject.toml`, `config.py`; `uv sync --extra offline --extra embed --extra web`; smoke import.
2. **DB helpers**: thêm hàm lấy keyframe theo video/pk + đếm.
3. **Embed**: `ingest/embed.py` + `kis embed`; chạy thử `--limit 1`, kiểm tra shape/norm/sha; chạy 20 video.
4. **FAISS**: `index/build.py` + `index/store.py` + `kis index build`; assert counts.
5. **Search**: `search/engine.py` + `kis search`; đo latency và kiểm tra top-20.
6. **UI**: `web/app.py` + `static/index.html` + `kis serve`; test bằng `curl` + trình duyệt.
7. **Tests**: 3 file test mới (không tải model): normalize/shard roundtrip, build+search id-map với vector tổng hợp, format kết quả.
8. **Đo & cập nhật tài liệu**: ghi throughput embed, latency search, kích thước artifact vào `README.md` §8; thêm `docs/Plan-Embed-FAISS-UI.md`.

## 8. Verification

```bash
# 1. Cài đặt + import
uv sync --extra offline --extra embed --extra web
uv run python -c "import torch, faiss, transformers, fastapi; print(torch.__version__, torch.version.cuda, faiss.__version__)"

# 2. Embed 20 video (resume được)
uv run kis embed --batch 32
# kỳ vọng: 14.061 ảnh, 20 shard; kiểm tra 1 shard:
uv run python - <<'PY'
import numpy as np, json, glob
p = sorted(glob.glob("index/embeddings/*/*.npy"))[0]
a = np.load(p)
print(p, a.shape, a.dtype, float(np.linalg.norm(a.astype("float32"), axis=1).mean()))
PY

# 3. Build FAISS + manifest
uv run kis index build
# kỳ vọng: ntotal == 14061; in nvideos/nkeyframes/partial

# 4. Search CLI
uv run kis search "người dẫn chương trình trong trường quay" --k 20
uv run kis search "..." --k 20 --json | head

# 5. UI
uv run kis serve --port 8000
curl -s localhost:8000/api/health
curl -s "localhost:8000/api/search?q=hoi+nghi&k=5"
curl -sI -H "Range: bytes=0-1023" localhost:8000/api/video/L13_V001 | head
# mở http://127.0.0.1:8000, thử search, click card, mở video

# 6. Test
uv run python -m unittest discover -s tests -v

# 7. Kiểm tra toàn vẹn: xoá tạm 1 shard rồi build phải fail
```

**Định nghĩa hoàn thành**
- `kis index build` pass với `ntotal == 14.061`.
- Search top-20 trả kết quả trong < 2 s p95 trên CPU (kỳ vọng < 300 ms).
- UI search được, xem được keyframe, video seek đúng `t`.
- Test mới + 15 test cũ pass.
- `README.md` có số đo thật; `docs/Plan-Embed-FAISS-UI.md` tồn tại.

## 9. Rủi ro & đối sách

| Rủi ro | Đối sách |
|---|---|
| torch kéo wheel CUDA | Dùng `pytorch-cpu` index; kiểm tra `torch.version.cuda is None` |
| `faiss-cpu` lệch ABI với numpy 2.5 | Verify `import faiss`; nếu lỗi nâng `faiss-cpu>=1.11` hoặc hạ numpy |
| SigLIP-2 hiểu tiếng Việt chưa tốt | Model đa ngữ; đo bằng 10–20 query thật; nếu kém mới cân nhắc Jina CLIP v2/translate ở bước sau |
| Keyframe pk có gap | `IndexIDMap2` + id tường minh; assert set(id) == set(pk) lúc build |
| Shard cũ sau `--force` | Sidecar có `keyframe_pks` + sha256; build fail nếu lệch DB; `kis embed --force` |
| Index/DB lệch âm thầm | Assert `ntotal == manifest.ntotal` và (strict) `== COUNT(keyframes)` khi load |
| Video không seek được | `FileResponse` + kiểm tra HTTP Range bằng curl |
| Model 0.4B tải chậm/lần đầu | Cache HuggingFace; log rõ khi download; embed resume |

## 10. Ngoài phạm vi (việc tiếp theo)

- OCR (RapidOCR) + ASR (faster-whisper) + FTS5 kép.
- Weighted RRF, temporal aggregation, Gemini structuring/rerank.
- `eval/queries.jsonl` + recall@20/MRR harness.
- Pin tray, timer/EV, DRES client + mock.
- Chạy embed full 363 video / toàn dataset, HNSW nếu p95 vượt ngân sách.
