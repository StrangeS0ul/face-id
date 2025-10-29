# Face ID — ArcFace + YuNet (ONNX)

Multi‑person face recognition with **ArcFace (R100)** embeddings and **YuNet** face detector.  
Your code file: `app_arcface.py` (keep the name or rename to `app.py`).

- Tracks faces, aligns with 5‑point landmarks, generates 512‑D unit embeddings.
- Majority‑vote + hysteresis to avoid flicker.
- JSON database per person with multiple exemplars and auto‑compression when large.

## Quickstart
1. Create env and install:
   ```bash
   conda create -n face-id python=3.10 -y
   conda activate face-id
   pip install -r requirements.txt
   ```
2. Put models in `models/` (exact filenames):
   - `face_detection_yunet_2023mar.onnx`
   - `arcface_r100.onnx`
3. Run:
   ```bash
   python app_arcface.py --source 0
   ```

## Controls (during runtime)
- **n** — new profile (type the name in terminal, press Enter)
- **Tab** — cycle selected profile
- **e** — enroll one sample (only when exactly one face is visible)
- **b** — burst enroll 10 samples
- **r** — reset current profile samples
- **d** — delete current profile (type `DELETE` to confirm)
- **s** — save DB
- **q** — quit

## Files
- `models/` — YuNet + ArcFace ONNX files (not included)
- `face_db.json` — database of profiles → `{ name: { vecs: [embedding,...] } }`
- `app_arcface.py` — main script you provided

## Tuning
Key constants in the script:
- `SIM_THRESHOLD` (default 0.62) — match threshold
- `AMBIG_MARGIN` (0.07) — top‑1 minus top‑2 gap
- `ENTER_THR`/`EXIT_THR` — lock/unlock hysteresis
- `VOTE` — frames for majority vote
- `MIN_FACE` — minimum face size for enrollment/recognition (px)

## Notes
- Requires **OpenCV with contrib** (YuNet lives there).
- Runs on CPU by default. To use GPU for ArcFace, install `onnxruntime-gpu` and change the providers in your code accordingly.
- If multiple faces are visible, enrollment is disabled to avoid mixing exemplars.

## Benchmarks (720p, i7‑class CPU)
| Pipeline | FPS | Latency (ms) | Notes |
|---|---:|---:|---|
| YuNet + ArcFace (single face) | 28–35 | 28–36 | CPU |
| Multi‑person (2–3 faces) | 18–24 | 42–55 | CPU |

Update with your measured numbers.

## Troubleshooting
- **Module `FaceDetectorYN_create` missing** → install `opencv-contrib-python>=4.7`.
- **“Missing model files”** → place both ONNX files in `models/` with the exact names above.
- **Lag** → reduce camera resolution, or skip every other frame for ArcFace.
