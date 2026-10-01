# NRMS - Real data

Pipeline PyTorch triển khai Neural News Recommendation with Multi-Head
Self-Attention (NRMS) trên dữ liệu log thật (title tiếng Việt). Code dùng lại
toàn bộ pipeline của project NRMS (MINDsmall); chỉ phần đọc dữ liệu và chia
split được thay đổi cho định dạng CSV mới.

Mô hình dùng title của news và word vector fastText tiếng Việt `cc.vi.300.vec`
(300 chiều, https://fasttext.cc/docs/en/crawl-vectors.html). Token khớp chính
xác được ưu tiên; nếu không có thì dùng vector của dạng viết hoa (ví dụ `Hà` cho
`hà`). Các từ còn lại được khởi tạo ngẫu nhiên và học trong quá trình train.
Loader vẫn đọc được file định dạng GloVe (không header) qua `input.word_vectors`.

## Cài đặt

Từ project root:

```powershell
python -m pip install -r requirement.txt
```

## Dữ liệu

```text
data/raw/
  behaviors/
    train_behaviors.csv
    val_behaviors.csv
    test_behaviors.csv
  news/
    train_news.csv
    val_news.csv
    test_news.csv
  fasttext/
    cc.vi.300.vec
```

Các file CSV **không có header**:

- `*_behaviors.csv`: `user_id,time,history,impressions`
  - `time`: `YYYY-MM-DD HH:MM:SS`
  - `history`: các news ID cách nhau bởi dấu cách, theo thứ tự thời gian
  - `impressions`: `NEWS_ID-1` (click) / `NEWS_ID-0` (không click), cách nhau bởi dấu cách
- `*_news.csv`: `news_id,title`

Khác biệt so với MIND khi preprocess:

- Không có cột impression ID nên ID được sinh dạng `<split>-<số dòng>`
  (ví dụ `train-12`, `test-305`).
- Validation và test lấy trực tiếp từ `val_*` và `test_*`, không chia dev theo
  thời gian như MIND.
- Log thật lặp lại cùng một news: candidate trùng trong một impression được gộp
  (nhãn = 1 nếu có ít nhất một lần click). `sequence.deduplicate_history: true`
  bỏ news trùng trong history (giữ lần xuất hiện gần nhất).
- Impression không có negative bị bỏ ở cả train/validation/test (không tính
  được AUC); số lượng bị bỏ được ghi trong log và `statistics.json`.

## Chạy pipeline

### 1. Preprocess

```powershell
python -m scripts.preprocess --config configs/preprocess.yaml
python -m scripts.validate_processed `
  --data-dir data/processed/real_nrms_fasttext_vi
```

Thêm `--overwrite` vào lệnh preprocess khi cần tạo lại processed dataset.

### 2. Train

```powershell
python -m scripts.train --config configs/train.yaml
```

Mỗi run được ghi vào:

```text
outputs/<experiment_name>/YYYY-MM-DD_HH-MM-SS/
```

### 3. Evaluate test set

```powershell
python -m scripts.evaluate `
  --run-dir outputs/real_nrms_fasttext_vi/YYYY-MM-DD_HH-MM-SS
```

Kết quả gồm test metrics và prediction cho từng impression/news.

### 4. Recommend

```powershell
python -m scripts.recommend `
  --run-dir outputs/real_nrms_fasttext_vi/YYYY-MM-DD_HH-MM-SS `
  --history 188260907102710952 188260908163714249 `
  --top-k 10
```

Giới hạn catalog bằng `--candidates <NEWS_ID> ...`. Thêm
`--output recommendations.json` để ghi kết quả ra file. CLI tự loại các news
đã xuất hiện trong history.

## Smoke test

[configs/train_smoke.yaml](configs/train_smoke.yaml) là cấu hình nhẹ để kiểm tra toàn bộ
đường train/validation trên processed dataset hiện có:

```powershell
python -m scripts.train `
  --config configs/train_smoke.yaml `
  --max-train-batches 1 `
  --max-validation-batches 1
```

Hai giới hạn batch là tham số CLI. Nếu bỏ chúng, config vẫn chạy toàn bộ một
epoch.

Processed dataset hiện dùng embedding 300 chiều, vì vậy
`model.embedding_dim` phải bằng `300`, kể cả khi dùng model nhỏ để smoke test.

## Resume training

Tăng `training.epochs` trong config tới tổng số epoch mong muốn, sau đó chạy:

```powershell
python -m scripts.train `
  --config configs/train.yaml `
  --resume outputs/real_nrms_fasttext_vi/YYYY-MM-DD_HH-MM-SS/checkpoints/last.pt
```

Resume tiếp tục trong run directory cũ và khôi phục model, optimizer, scheduler,
history, best metric cùng trạng thái early stopping. Model và training config
phải khớp checkpoint; chỉ tổng số epoch được phép tăng.

## Output chính

```text
outputs/<experiment>/<timestamp>/
  checkpoints/
    best.pt
    last.pt
  artifacts/
    summary.json
    test_metrics.json
  plots/
    loss.png
    auc.png
    mrr.png
    ndcg.png
  predictions/
    test_predictions.csv
  config.yaml
  history.json
  run_info.json
  train.log
```

Validation và test báo cáo AUC, MRR, nDCG@5 và nDCG@10. News vectors được
precompute một lần mỗi evaluation pass để tránh encode lặp lại cùng news.

## Cấu hình tùy chọn

- `training.deterministic: true`: bật deterministic mode của PyTorch/CUDA.
- `loader.num_workers > 0`: worker được seed để negative sampling tái lập.
- `scheduler.type`: `none`, `reduce_on_plateau` hoặc `cosine`.
- `device`: `auto`, `cpu` hoặc `cuda`.