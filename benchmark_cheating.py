"""
Benchmark Script: YOLOv8 Segmentation
Membandingkan performa model .pt (original) vs .onnx (quantized/optimized)
Jetson Device - CPU Inference
"""

import time
import numpy as np
import cv2
import psutil
import os
import json
from datetime import datetime
from pathlib import Path

# ── Konfigurasi ──────────────────────────────────────────────────────────────
PT_MODEL_PATH   = "/home/jetson/cheating/models/best_yolov8n_seg_cheating.pt"
ONNX_MODEL_PATH = "/home/jetson/cheating/models/best_int8.onnx"
OUTPUT_DIR      = "/home/jetson/cheating/benchmark_results"
IMG_SIZE        = 640
NUM_WARMUP      = 5       # iterasi warmup (tidak dihitung)
NUM_RUNS        = 50      # iterasi benchmark utama
SOURCE          = None    # None = pakai dummy image; isi path video/gambar jika mau

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_dummy_frame(size=640):
    """Buat frame dummy untuk benchmark tanpa sumber eksternal."""
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_file_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024

def benchmark_pytorch(model_path, num_warmup, num_runs, source=None):
    """Benchmark model PyTorch (.pt)"""
    print("\n" + "="*60)
    print("📦 BENCHMARK: PyTorch (.pt) — Original Model")
    print("="*60)

    from ultralytics import YOLO
    model = YOLO(model_path, task="segment")

    frame = get_dummy_frame(IMG_SIZE) if source is None else cv2.imread(source)

    # Warmup
    print(f"  🔥 Warmup ({num_warmup} iterasi)...")
    for _ in range(num_warmup):
        model.predict(frame, imgsz=IMG_SIZE, device="cpu", verbose=False)

    # Benchmark
    print(f"  ⏱️  Benchmark ({num_runs} iterasi)...")
    mem_before = get_memory_mb()
    latencies = []

    for i in range(num_runs):
        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=IMG_SIZE, device="cpu", verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{num_runs}] latency: {latencies[-1]:.1f} ms")

    mem_after = get_memory_mb()

    stats = compute_stats(latencies, model_path, mem_before, mem_after)
    print_stats(stats, "PyTorch (.pt)")
    return stats

def benchmark_onnx(model_path, num_warmup, num_runs, source=None):
    """Benchmark model ONNX (.onnx)"""
    print("\n" + "="*60)
    print("⚡ BENCHMARK: ONNX — Optimized Model")
    print("="*60)

    from ultralytics import YOLO
    model = YOLO(model_path, task="segment")

    frame = get_dummy_frame(IMG_SIZE) if source is None else cv2.imread(source)

    # Warmup
    print(f"  🔥 Warmup ({num_warmup} iterasi)...")
    for _ in range(num_warmup):
        model.predict(frame, imgsz=IMG_SIZE, device="cpu", verbose=False)

    # Benchmark
    print(f"  ⏱️  Benchmark ({num_runs} iterasi)...")
    mem_before = get_memory_mb()
    latencies = []

    for i in range(num_runs):
        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=IMG_SIZE, device="cpu", verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{num_runs}] latency: {latencies[-1]:.1f} ms")

    mem_after = get_memory_mb()

    stats = compute_stats(latencies, model_path, mem_before, mem_after)
    print_stats(stats, "ONNX (.onnx)")
    return stats

def compute_stats(latencies, model_path, mem_before, mem_after):
    latencies = np.array(latencies)
    return {
        "model_path"    : model_path,
        "file_size_mb"  : round(get_file_size_mb(model_path), 2),
        "memory_used_mb": round(mem_after - mem_before, 2),
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "latency_min_ms" : round(float(np.min(latencies)), 2),
        "latency_max_ms" : round(float(np.max(latencies)), 2),
        "latency_p50_ms" : round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms" : round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms" : round(float(np.percentile(latencies, 99)), 2),
        "fps_mean"       : round(1000 / float(np.mean(latencies)), 2),
        "num_runs"       : len(latencies),
        "all_latencies_ms": latencies.tolist(),
    }

def print_stats(stats, label):
    print(f"\n  📊 Hasil [{label}]:")
    print(f"     File size      : {stats['file_size_mb']} MB")
    print(f"     Memory used    : {stats['memory_used_mb']} MB")
    print(f"     Latency mean   : {stats['latency_mean_ms']} ms")
    print(f"     Latency min    : {stats['latency_min_ms']} ms")
    print(f"     Latency max    : {stats['latency_max_ms']} ms")
    print(f"     Latency P50    : {stats['latency_p50_ms']} ms")
    print(f"     Latency P95    : {stats['latency_p95_ms']} ms")
    print(f"     Latency P99    : {stats['latency_p99_ms']} ms")
    print(f"     FPS (avg)      : {stats['fps_mean']}")

def print_comparison(pt_stats, onnx_stats):
    print("\n" + "="*60)
    print("📈 PERBANDINGAN: PT vs ONNX")
    print("="*60)

    speedup     = pt_stats['latency_mean_ms'] / onnx_stats['latency_mean_ms']
    size_ratio  = pt_stats['file_size_mb'] / onnx_stats['file_size_mb']
    fps_gain    = onnx_stats['fps_mean'] - pt_stats['fps_mean']

    print(f"\n  {'Metrik':<25} {'PT (original)':<20} {'ONNX (optimized)':<20} {'Selisih'}")
    print(f"  {'-'*80}")
    print(f"  {'File Size (MB)':<25} {pt_stats['file_size_mb']:<20} {onnx_stats['file_size_mb']:<20} {size_ratio:.2f}x")
    print(f"  {'Latency Mean (ms)':<25} {pt_stats['latency_mean_ms']:<20} {onnx_stats['latency_mean_ms']:<20} {speedup:.2f}x faster" if speedup >= 1 else f"  {'Latency Mean (ms)':<25} {pt_stats['latency_mean_ms']:<20} {onnx_stats['latency_mean_ms']:<20} {1/speedup:.2f}x slower")
    print(f"  {'Latency P95 (ms)':<25} {pt_stats['latency_p95_ms']:<20} {onnx_stats['latency_p95_ms']:<20}")
    print(f"  {'FPS (avg)':<25} {pt_stats['fps_mean']:<20} {onnx_stats['fps_mean']:<20} +{fps_gain:.2f} FPS")
    print(f"  {'Memory Used (MB)':<25} {pt_stats['memory_used_mb']:<20} {onnx_stats['memory_used_mb']:<20}")

    print(f"\n  ✅ Speedup      : {speedup:.2f}x {'lebih cepat' if speedup >= 1 else 'lebih lambat'} (ONNX vs PT)")
    print(f"  ✅ Size ratio   : {size_ratio:.2f}x (PT lebih besar dari ONNX)")
    print(f"  ✅ FPS gain     : +{fps_gain:.2f} FPS dengan ONNX")

def save_results(pt_stats, onnx_stats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp"  : timestamp,
        "config"     : {"img_size": IMG_SIZE, "num_runs": NUM_RUNS, "num_warmup": NUM_WARMUP},
        "pt_model"   : pt_stats,
        "onnx_model" : onnx_stats,
        "comparison" : {
            "speedup_x"       : round(pt_stats['latency_mean_ms'] / onnx_stats['latency_mean_ms'], 3),
            "size_reduction_x": round(pt_stats['file_size_mb'] / onnx_stats['file_size_mb'], 3),
            "fps_gain"        : round(onnx_stats['fps_mean'] - pt_stats['fps_mean'], 2),
        }
    }

    json_path = f"{OUTPUT_DIR}/benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Simpan juga summary .txt
    txt_path = f"{OUTPUT_DIR}/benchmark_{timestamp}_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"BENCHMARK SUMMARY — {timestamp}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Config: imgsz={IMG_SIZE}, runs={NUM_RUNS}, warmup={NUM_WARMUP}\n\n")
        f.write(f"{'Metrik':<25} {'PT':<15} {'ONNX':<15}\n")
        f.write(f"{'-'*55}\n")
        for key in ['file_size_mb','memory_used_mb','latency_mean_ms',
                    'latency_p50_ms','latency_p95_ms','latency_p99_ms','fps_mean']:
            f.write(f"{key:<25} {str(pt_stats[key]):<15} {str(onnx_stats[key]):<15}\n")
        f.write(f"\nSpeedup  : {result['comparison']['speedup_x']}x\n")
        f.write(f"Size Red : {result['comparison']['size_reduction_x']}x\n")
        f.write(f"FPS Gain : +{result['comparison']['fps_gain']}\n")

    print(f"\n  💾 Hasil disimpan ke:")
    print(f"     JSON   : {json_path}")
    print(f"     Summary: {txt_path}")

    return json_path, txt_path

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 YOLO Segmentation Benchmark — PT vs ONNX")
    print(f"   Device  : Jetson (CPU)")
    print(f"   Img size: {IMG_SIZE}")
    print(f"   Runs    : {NUM_RUNS} (+ {NUM_WARMUP} warmup)")

    pt_stats   = benchmark_pytorch(PT_MODEL_PATH, NUM_WARMUP, NUM_RUNS, SOURCE)
    onnx_stats = benchmark_onnx(ONNX_MODEL_PATH, NUM_WARMUP, NUM_RUNS, SOURCE)

    print_comparison(pt_stats, onnx_stats)
    save_results(pt_stats, onnx_stats)

    print("\n✅ Benchmark selesai!\n")