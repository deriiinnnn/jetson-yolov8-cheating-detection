# Jetson YOLOv8 Cheating Detection

Implementasi model **YOLOv8 Segmentation** untuk deteksi kecurangan ujian berbasis **Edge AI** pada perangkat **NVIDIA Jetson**. Project ini berisi model hasil training, script inference gambar, script realtime camera/IP camera, serta benchmark performa model `.pt` dan `.onnx`.

## Fitur Utama

- Deteksi objek dengan kelas `cheating` dan `normal`.
- Inference gambar menggunakan model YOLOv8 segmentation.
- Realtime detection menggunakan kamera HP/IP camera.
- Benchmark performa model PyTorch `.pt` dan ONNX `.onnx`.
- Output hasil deteksi disimpan ke folder `result/`.
- Model tersimpan di folder `models/`.

## Struktur Folder

```text
.
├── models/
│   ├── best_yolov8n_seg_cheating.pt
│   ├── best_yolov8n_seg_cheating.onnx
│   └── best_int8.onnx
├── benchmark_results/
│   ├── benchmark_20260623_211646.json
│   └── benchmark_20260623_211646_summary.txt
├── result/
│   ├── cheating.jpg
│   ├── cheating/
│   └── normal/
├── predict/
│   └── test.jpg
├── benchmark_cheating.py
├── realtime_cheating.py
├── test.py
├── test1.py
├── test2.py
├── test_camera.py
├── camera.py
└── test.jpg
```

## Kebutuhan Sistem

Project ini ditujukan untuk perangkat Jetson atau komputer lokal yang mendukung Python dan OpenCV.

Dependency utama:

```bash
pip install -r requirements.txt
```

Isi dependency utama:

- `ultralytics`
- `opencv-python`
- `numpy`
- `psutil`
- `onnxruntime`

## Cara Menjalankan Inference Gambar

Pastikan model berada di folder `models/`, lalu jalankan:

```bash
python3 test.py
```

Script tersebut akan membaca gambar `test.jpg`, melakukan prediksi, lalu menyimpan hasil ke folder `result/`.

## Cara Menjalankan Realtime Detection

Edit bagian URL kamera pada file `realtime_cheating.py`:

```python
url = "http://IP_KAMERA:PORT/video"
```

Lalu jalankan:

```bash
python3 realtime_cheating.py
```

Tekan tombol `ESC` untuk menghentikan program.

## Cara Menjalankan Benchmark

File benchmark membandingkan performa model `.pt` dan `.onnx`.

```bash
python3 benchmark_cheating.py
```

Hasil benchmark akan disimpan otomatis ke folder `benchmark_results/` dalam format `.json` dan `.txt`.

## Hasil Benchmark

Benchmark dilakukan dengan konfigurasi `imgsz=640`, `runs=50`, dan `warmup=5`.

| Metrik | PyTorch `.pt` | ONNX INT8 |
|---|---:|---:|
| File size | 6.48 MB | 3.52 MB |
| Memory used | 14.33 MB | 1.01 MB |
| Mean latency | 854.68 ms | 81.05 ms |
| P50 latency | 853.04 ms | 78.75 ms |
| P95 latency | 869.90 ms | 90.61 ms |
| FPS average | 1.17 FPS | 12.34 FPS |

ONNX INT8 menghasilkan peningkatan performa sekitar **10.545x lebih cepat** dibandingkan model `.pt` pada pengujian ini.

## Catatan

- Jika menjalankan di Jetson, sesuaikan path model jika project tidak berada di `/home/jetson/cheating/`.
- Jika kamera tidak terbaca, pastikan IP camera/HP dan Jetson berada di jaringan Wi-Fi yang sama.
- Untuk model yang lebih besar dari 100 MB, gunakan Git LFS agar bisa diunggah ke GitHub.

## Author

Bella Adisty
