# Face ID — ArcFace + YuNet (ONNX)

Multi‑person face recognition on the webcam. **YuNet** detects faces and 5 landmarks; **ArcFace R100** produces 512‑D unit embeddings; a small temporal filter (majority voting + hysteresis) stabilizes identity over time.

- CPU works out of the box. CUDA if `onnxruntime-gpu` is installed.
- Model files are tracked with **Git LFS** so fresh clones include them automatically.

---

## Quickstart

```bash
# new env (recommended) or reuse an existing one with Python ≥3.10
conda create -n face-id python=3.10 -y
conda activate face-id
pip install -r requirements.txt
```

Run:
```bash
python app_arcface.py --source 0
```

Expected performance at 1280×720, CPU laptop:
- single face: **28–35 FPS**, ~**28–36 ms** latency
- two–three faces: **18–24 FPS**, ~**42–55 ms** latency

GPU raises FPS if you install `onnxruntime-gpu`.

---

## Controls

- **n**: create a new profile (type name, Enter)
- **Tab**: switch selected profile
- **e**: enroll one sample for the selected profile
- **b**: burst enroll 10 samples
- **r**: reset samples for the selected profile
- **d**: delete selected profile (type `DELETE` to confirm)
- **s**: save database to `face_db.json`
- **q**: quit

The database stores per‑profile embedding arrays for fast cosine matching.

---

## How it works

1) **Detect + align** → YuNet box + 5 landmarks → similarity transform to 112×112.  
2) **Embed** → ArcFace R100 ONNX → L2‑normalized 512‑D vector.  
3) **Match** → cosine similarity vs enrolled exemplars → majority vote over a short window with enter/exit thresholds to avoid ID flicker.

Key thresholds (in code):
- `SIM_THRESHOLD = 0.62` (match floor)
- `AMBIG_MARGIN = 0.07`  (top‑1 minus top‑2 margin)
- `ENTER_THR` / `EXIT_THR` (hysteresis)
- `VOTE` (temporal window)
- `MIN_FACE` (min face size in px for enrollment/ID)

---

## Models (LFS)

These ONNX files are stored in `models/` and pulled by Git LFS:
- `models/face_detection_yunet_2023mar.onnx`
- `models/arcface_r100.onnx`

If a fresh clone shows pointer files, run:
```bash
git lfs install
git lfs pull
```

---

## Requirements

```
numpy>=1.23.0
opencv-contrib-python>=4.7.0
onnxruntime>=1.17.0    # or onnxruntime-gpu for CUDA
```

---

## License

MIT
