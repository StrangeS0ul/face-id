import os, sys, json, time, argparse, itertools, collections
import numpy as np
import cv2 as cv
import onnxruntime as ort

# ---- paths ----
ROOT   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
DB_PATH = os.path.join(ROOT, "face_db.json")

YUNET   = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")   # detector + 5 landmarks
ARCFACE = os.path.join(MODELS, "arcface_r100.onnx")                    # recognizer (expects 112x112)

# ---- thresholds & params ----
SIM_THRESHOLD = 0.62        # 0.60–0.70 typical
AMBIG_MARGIN  = 0.07        # top1-top2 similarity gap
ENTER_THR     = 0.64        # temporal lock enter
EXIT_THR      = 0.58        # temporal lock exit
VOTE          = 7           # frames for majority vote
MIN_FACE      = 72          # min bbox side in px to recognize/enroll
MAX_VECS_PER_PROFILE = 60   # cap exemplars; we also compress when >50

TRACK_IOU_THR  = 0.30       # IoU association threshold
TRACK_MAX_MISS = 12         # frames to keep an unassigned track

# five-point ArcFace reference (112x112)
ARC_SRC5 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# ---------- utils ----------
def ensure_models():
    miss = []
    if not os.path.exists(YUNET):   miss.append(YUNET)
    if not os.path.exists(ARCFACE): miss.append(ARCFACE)
    if miss:
        print("Missing model files:"); [print("-", m) for m in miss]
        sys.exit(1)

def load_db():
    if not os.path.exists(DB_PATH): return {"profiles": {}}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)
    # migrate old schema {emb,count} -> {vecs:[unit_vec,...]}
    for name, pdata in list(db.get("profiles", {}).items()):
        if "vecs" not in pdata:
            vecs = []
            emb = pdata.get("emb")
            if emb is not None:
                v = np.asarray(emb, np.float32).reshape(-1)
                v /= (np.linalg.norm(v) + 1e-9)
                vecs = [v.tolist()]
            db["profiles"][name] = {"vecs": vecs}
    return db

def save_db(db):
    tmp = DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp, DB_PATH)

def estimate_norm5(lmk5, image_size=112):
    dst = ARC_SRC5.copy()
    if image_size != 112:
        s = image_size / 112.0
        dst *= s
    M, _ = cv.estimateAffinePartial2D(lmk5, dst, method=cv.LMEDS)
    return M

def align_crop_112(frame, pts5):
    M = estimate_norm5(pts5.astype(np.float32), 112)
    return cv.warpAffine(frame, M, (112, 112), flags=cv.INTER_LINEAR)

def arcface_embed(sess, img112):
    # BGR -> RGB, normalize to [-1,1]
    x = cv.cvtColor(img112, cv.COLOR_BGR2RGB).astype(np.float32)
    x = (x - 127.5) / 128.0
    inp = sess.get_inputs()[0]
    shape = inp.shape  # [1,112,112,3] or [1,3,112,112]
    if len(shape) == 4 and shape[1] == 3:  # NCHW
        x = np.transpose(x, (2, 0, 1))[None, ...]
    else:                                   # NHWC
        x = x[None, ...]
    out = sess.run(None, {inp.name: x})[0]
    v = out[0].astype(np.float32).reshape(-1)
    v /= (np.linalg.norm(v) + 1e-9)
    return v  # 512-D unit vector for ArcFace-R100

def best_match(db, feat_u):
    best_name, best_sim, second_sim = "Unknown", -1.0, -1.0
    for name, pdata in db["profiles"].items():
        vecs = np.asarray(pdata.get("vecs", []), dtype=np.float32)
        if vecs.size == 0: continue
        if vecs.ndim != 2 or vecs.shape[1] != feat_u.size:  # skip mismatched dims
            continue
        sims = vecs @ feat_u
        s = float(np.max(sims))
        if s > best_sim:
            second_sim = best_sim
            best_sim = s; best_name = name
        elif s > second_sim:
            second_sim = s
    if best_sim < SIM_THRESHOLD or (best_sim - max(second_sim, -1.0)) < AMBIG_MARGIN:
        return "Unknown", best_sim, second_sim
    return best_name, best_sim, second_sim

def selected_count(db, name):
    try: return len(db["profiles"][name]["vecs"])
    except: return 0

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    iw = max(0, x2-x1); ih = max(0, y2-y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = aw*ah + bw*bh - inter
    return inter / ua

# ---------- per-face track ----------
_track_id_gen = itertools.count(1)

class Track:
    def __init__(self, bbox, lmk5=None):
        self.id   = next(_track_id_gen)
        self.bbox = bbox            # (x,y,w,h)
        self.lmk5 = lmk5            # 5x2 landmarks for current frame
        self.miss = 0
        self.hist = collections.deque(maxlen=VOTE)
        self.lock = None            # locked label via hysteresis

    def vote(self, label, score, enter=ENTER_THR, exit=EXIT_THR):
        self.hist.append(label if (label != "Unknown" and score >= enter) else "Unknown")
        if self.hist:
            top = max(set(self.hist), key=self.hist.count)
            freq = self.hist.count(top)
        else:
            top, freq = "Unknown", 0
        if self.lock is None and top != "Unknown" and freq >= (VOTE//2 + 1):
            self.lock = top
        elif self.lock is not None and (label == "Unknown" or score < exit):
            self.lock = None
        return self.lock or label

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video path")
    args = ap.parse_args()
    ensure_models()

    # detector (YuNet)
    det = cv.FaceDetectorYN_create(YUNET, "", (320, 240), score_threshold=0.85, nms_threshold=0.3, top_k=5000)
    # recognizer (ArcFace)
    sess = ort.InferenceSession(ARCFACE, providers=["CPUExecutionProvider"])

    # capture
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv.VideoCapture(src, cv.CAP_DSHOW if isinstance(src, int) else 0)
    if not cap.isOpened():
        print("Camera open failed"); sys.exit(1)
    ok, frame = cap.read()
    if not ok:
        print("No frames from source"); sys.exit(1)
    h, w = frame.shape[:2]
    det.setInputSize((w, h))

    db = load_db()
    names = sorted(db["profiles"].keys()); sel_idx = 0 if names else -1
    selected = names[sel_idx] if sel_idx >= 0 else None

    print("Keys: [n]=new  [e]=enroll  [b]=burst x10  [Tab]=next  [d]=delete  [r]=reset  [s]=save  [q]=quit")
    tracks = []
    last = time.time(); frames = 0; fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break

        dets = det.detect(frame)[1]
        faces = [] if dets is None else dets

        # --- associate detections to existing tracks by IoU ---
        assigned = set()
        for t in tracks:
            best_iou, best_j = -1.0, -1
            for j, f in enumerate(faces):
                if j in assigned: continue
                x, y, w0, h0 = map(int, f[:4])
                ov = iou(t.bbox, (x, y, w0, h0))
                if ov > best_iou:
                    best_iou, best_j = ov, j
            if best_j >= 0 and best_iou >= TRACK_IOU_THR:
                f = faces[best_j]; assigned.add(best_j)
                x, y, w0, h0 = map(int, f[:4])
                t.bbox = (x, y, w0, h0)
                t.lmk5 = np.array([[f[4],f[5]],[f[6],f[7]],[f[8],f[9]],[f[10],f[11]],[f[12],f[13]]], dtype=np.float32)
                t.miss = 0
            else:
                t.miss += 1

        # --- start new tracks for unassigned detections ---
        for j, f in enumerate(faces):
            if j in assigned: continue
            x, y, w0, h0 = map(int, f[:4])
            lmk5 = np.array([[f[4],f[5]],[f[6],f[7]],[f[8],f[9]],[f[10],f[11]],[f[12],f[13]]], dtype=np.float32)
            tracks.append(Track((x, y, w0, h0), lmk5=lmk5))

        # --- drop stale tracks ---
        tracks = [t for t in tracks if t.miss <= TRACK_MAX_MISS]

        # --- recognize each track ---
        for t in tracks:
            x, y, w0, h0 = map(int, t.bbox)
            if min(w0, h0) < MIN_FACE or t.lmk5 is None:
                cv.rectangle(frame, (x, y), (x+w0, y+h0), (0, 255, 0), 1)
                cv.putText(frame, f"#{t.id}", (x, y-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                continue

            crop = align_crop_112(frame, t.lmk5)
            vec  = arcface_embed(sess, crop)
            label, s1, s2 = best_match(db, vec)
            disp = t.vote(label, s1, enter=ENTER_THR, exit=EXIT_THR)

            cv.rectangle(frame, (x, y), (x+w0, y+h0), (0, 255, 0), 2)
            cv.putText(frame, f"{disp} ({s1:.2f})  id#{t.id}", (x, y-8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # HUD
        frames += 1
        now = time.time()
        if now - last >= 0.5:
            fps = frames / (now - last); frames = 0; last = now
        cnt = selected_count(db, selected) if selected else 0
        hud = f"Profiles:{len(db['profiles'])}  Selected:{selected or '-'}  Samples:{cnt}  FPS:{fps:.1f}  Thr:{SIM_THRESHOLD}  Mg:{AMBIG_MARGIN}"
        cv.putText(frame, hud, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        if len(faces) > 1:
            cv.putText(frame, "Multiple faces. Enrollment disabled.", (10, 46),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

        cv.imshow("Face ID - ArcFace (multi-person)", frame)
        k = cv.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        elif k == ord('n'):
            print("Type new profile name then Enter:"); cv.waitKey(1)
            try: name = input().strip()
            except EOFError: name = ""
            if name and name not in db["profiles"]:
                db["profiles"][name] = {"vecs": []}; save_db(db)
                names = sorted(db["profiles"].keys()); sel_idx = names.index(name); selected = name
                print(f"Created: {name}. Use 'e' or 'b' to enroll.")
            else:
                print("Invalid or exists.")

        elif k == ord('s'):
            save_db(db); print("Saved.")

        elif k in (9,):  # Tab
            if db["profiles"]:
                names = sorted(db["profiles"].keys())
                sel_idx = 0 if sel_idx < 0 else (sel_idx + 1) % len(names)
                selected = names[sel_idx]
                print(f"Selected: {selected} (samples={selected_count(db, selected)})")

        elif k == ord('d'):
            if selected and selected in db["profiles"]:
                print(f"Confirm delete '{selected}'. Type DELETE:"); cv.waitKey(1)
                try: resp = input().strip().upper()
                except EOFError: resp = ""
                if resp == "DELETE":
                    del db["profiles"][selected]; save_db(db)
                    names = sorted(db["profiles"].keys())
                    sel_idx = 0 if names else -1
                    selected = names[sel_idx] if sel_idx >= 0 else None
                    print("Deleted.")
                else:
                    print("Cancelled.")

        elif k == ord('r'):
            if selected and selected in db["profiles"]:
                db["profiles"][selected]["vecs"] = []; save_db(db)
                print(f"Reset '{selected}'.")

        elif k == ord('e'):
            # strict: exactly one face visible to avoid contaminating exemplars
            if not selected: print("No profile selected"); continue
            if len(faces) != 1: print("Enrollment requires exactly one face."); continue
            x, y, w0, h0 = map(int, faces[0][:4])
            if min(w0, h0) < MIN_FACE: print("Move closer for enrollment."); continue
            lmk = np.array([[faces[0][4],faces[0][5]],[faces[0][6],faces[0][7]],
                            [faces[0][8],faces[0][9]],[faces[0][10],faces[0][11]],
                            [faces[0][12],faces[0][13]]], dtype=np.float32)
            crop = align_crop_112(frame, lmk)
            vec  = arcface_embed(sess, crop)
            vecs = db["profiles"][selected].setdefault("vecs", [])
            vecs.append(vec.tolist())

            # compress when many samples to keep speed
            if len(vecs) > 50:
                X = np.asarray(vecs, dtype=np.float32)
                k = 16  # centroid count
                # k-means++ init (cosine)
                C = [X[np.random.randint(len(X))]]
                for _ in range(1, k):
                    d2 = np.max(1.0 - (X @ (np.asarray(C).T)), axis=1)
                    C.append(X[np.argmax(d2)])
                C = np.asarray(C, dtype=np.float32)
                for _ in range(10):  # Lloyd steps
                    sims = X @ C.T
                    idx = np.argmax(sims, axis=1)
                    for i in range(k):
                        grp = X[idx == i]
                        if len(grp):
                            c = grp.mean(axis=0)
                            C[i] = c / (np.linalg.norm(c) + 1e-9)
                db["profiles"][selected]["vecs"] = C.tolist()

            save_db(db)
            print(f"Enrolled -> {selected}. samples={len(db['profiles'][selected]['vecs'])}")

        elif k == ord('b'):
            if not selected: print("No profile selected"); continue
            if len(faces) != 1: print("Burst requires exactly one face."); continue
            count = 0; t0 = time.time()
            while count < 10:
                ok2, f2 = cap.read()
                if not ok2: break
                d2 = det.detect(f2)[1]
                if d2 is None or len(d2) != 1: continue
                x, y, w0, h0 = map(int, d2[0][:4])
                if min(w0, h0) < MIN_FACE: continue
                lmk = np.array([[d2[0][4],d2[0][5]],[d2[0][6],d2[0][7]],
                                [d2[0][8],d2[0][9]],[d2[0][10],d2[0][11]],
                                [d2[0][12],d2[0][13]]], dtype=np.float32)
                crop = align_crop_112(f2, lmk)
                vec  = arcface_embed(sess, crop)
                db["profiles"][selected].setdefault("vecs", []).append(vec.tolist())
                if len(db["profiles"][selected]["vecs"]) > MAX_VECS_PER_PROFILE:
                    db["profiles"][selected]["vecs"] = db["profiles"][selected]["vecs"][-MAX_VECS_PER_PROFILE:]
                count += 1
                while time.time() - t0 < 0.12: cv.waitKey(1)
                t0 = time.time()
            save_db(db)
            print(f"Burst {count} -> {selected}. total={len(db['profiles'][selected]['vecs'])}")

    cap.release(); cv.destroyAllWindows(); save_db(db)

if __name__ == "__main__":
    main()
