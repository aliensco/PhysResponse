"""Camera-motion compensation for generated videos (the inversion assumes the reference's fixed camera; many video models
add camera motion, which makes every object 'not explained'). Registrations are 2D affine maps of the BACKGROUND (tracked
objects' boxes, dilated, masked out; 427x240 grey), fitted by ECC and judged by the LOCAL CORRELATION of the band-passed
frames on the textured background. It ignores brightness and contrast, and an untracked object over part of the background
lowers it for the right map without rewarding a wrong one; the old floor on ECC correlation (0.85) dropped still cameras whose
background brightens or sharpens and accepted maps that follow a large untracked object (F113). Per tracked frame f, q the
previous one and p the last accepted one:
  neighbour  q -> f: ECC from identity, or identity, whichever correlates better at the fine scale BAND_SEL;
  chain      the neighbour map composed with q's map (accepted, else q's chain): the prediction of the 0 -> f map;
  direct     0 -> f: ECC started from the chain, then (if that does not do) from the last accepted map and from identity, and
             the identity itself; a rejected map never seeds a later frame.
The frame is accepted with the direct map (best mean local correlation at BAND_SEL) that agrees with the chain within TOL at
the tracked boxes and agrees with frame 0 (share of background pixels correlating >= LNCC at scale BAND) at least S_MIN, above
unrelated images (null test). After rejected frames the chain may have drifted: then also a direct map that agrees within TOL
with q's direct map carried by the neighbour map, stays within TRI_BASE + TRI_RATE (f - p) of the chain and agrees with frame
0 at least as well as the chain (an alias far from the chain, e.g. along horizontal edges under a pan, cannot pass).
Still camera: when every accepted map moves the tracked boxes less than STILL_PX, the video is not compensated (identity),
also on the frames whose neighbour chain agrees >= S_MIN at every step since the last accepted frame and stays within STILL_PX. Each object box at frame f is mapped into frame-0 coordinates
(box of the 4 mapped corners); frames without an accepted map keep NaN boxes. Output: tracks in the same format with mode
'lean+camcomp' and per-video statistics (max_shift_px: largest shift of the image centre over the accepted frames, as
measured; max_corner_px: largest displacement of the image corners, which a zoom moves too (F120); frames_failed: frames
without an accepted map; still; size: video width and height).
Usage: python cam_comp.py ITEMS_JSON TRACKS.jsonl [...] --out OUT.jsonl [--workers 9] [--keys KEYFILE]"""
import argparse, collections, json
from multiprocessing import get_context
import cv2
import numpy as np

SZ = (427, 240)
TOL = 2.0        # px of the 427x240 image (0.5 % of the width): largest disagreement of two maps at the tracked boxes
TRI_BASE, TRI_RATE, TRI_MAX = 2.0, 1.0, 15.0   # re-anchoring after rejected frames: at most this far from the chain (+ per frame since p)
STILL_PX = 2.0   # px of the video: a camera that moves the tracked boxes less than this on every accepted frame is not compensated
BAND = 5         # px of the 427x240 image: scale of the structure (band-pass and local-correlation window) for the acceptance score
BAND_SEL = 2     # ... and the finer scale that chooses between maps (sensitive to sub-pixel misalignment)
LNCC = 0.6       # a background pixel agrees when the local correlation of the band-passed images reaches this
S_MIN = 0.2      # least agreement (scale BAND) of an accepted map; unrelated images: 99th percentile 0.18 (F113 null test)
MIN_PIX = 500    # least textured background pixels for a score
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
I3 = np.eye(3)
YS, XS = np.mgrid[0:SZ[1], 0:SZ[0]].astype(np.float32)


def mask_for(boxes, W, H):
    m = np.full((SZ[1], SZ[0]), 255, np.uint8); sx, sy = SZ[0] / W, SZ[1] / H
    for b in boxes:
        if b is None or np.isnan(b[0]): continue
        x0, y0, x1, y1 = b; pad = 0.15 * max(x1 - x0, y1 - y0) + 6
        cv2.rectangle(m, (int((x0 - pad) * sx), int((y0 - pad) * sy)), (int((x1 + pad) * sx), int((y1 + pad) * sy)), 0, -1)
    return m


def bandpass(g, band=BAND):
    """structure of a grey image without its brightness and contrast trends"""
    return g - cv2.GaussianBlur(g, (0, 0), band)


def textured(d, band=BAND):
    return np.sqrt(cv2.GaussianBlur(d * d, (0, 0), band)) > 0.01


def warped(img, M, band=BAND):
    """band-passed image of img seen through the t -> f map M in t's coordinates (NaN where M leaves img)"""
    w = bandpass(cv2.warpAffine(img, M[:2].astype(np.float32), SZ, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE), band)
    u = M[0, 0] * XS + M[0, 1] * YS + M[0, 2]; v = M[1, 0] * XS + M[1, 1] * YS + M[1, 2]
    w[(u < 6) | (u > SZ[0] - 7) | (v < 6) | (v > SZ[1] - 7)] = np.nan
    return w


def local_corr(a, w, valid, band=BAND):
    """local correlation (Gaussian window of the band's scale) of the band-passed images a and w (NaN outside the frame) at
    the valid pixels; None when fewer than MIN_PIX"""
    ok = valid & ~np.isnan(w)
    if ok.sum() < MIN_PIX: return None
    a0 = np.where(ok, a, 0).astype(np.float32); w0 = np.where(ok, w, 0).astype(np.float32)
    G = lambda x: cv2.GaussianBlur(x, (0, 0), band)
    return (G(a0 * w0) / np.sqrt(G(a0 * a0) * G(w0 * w0) + 1e-12))[ok]


def agreement(a, w, valid, band=BAND):
    """share of the valid pixels where the band-passed images correlate locally at least LNCC: unrelated images score ~0,
    the same view ~1 whatever its brightness and contrast"""
    r = local_corr(a, w, valid, band)
    return 0.0 if r is None else float((r >= LNCC).mean())


def ecc(tpl, img, init, msk):
    """3x3 matrix mapping template (small) coords -> image coords; None when ECC fails"""
    try: _, M = cv2.findTransformECC(tpl, img, init[:2].astype(np.float32), cv2.MOTION_AFFINE, CRIT, msk, 3)
    except cv2.error: return None
    return np.vstack([M.astype(float), [0, 0, 1]])


def gap(X, Y, P):
    """largest distance between the frame-0 positions that two 0 -> f maps give to the frame-f points P (small px)"""
    a = P @ np.linalg.inv(X).T; b = P @ np.linalg.inv(Y).T
    return float(np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]).max())


def _map_box(b, T):
    """box of the 4 corners of xyxy box b mapped by the 3x3 matrix T"""
    P = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[0], b[3], 1], [b[2], b[3], 1]]) @ T.T
    return [float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())]


def register(path, tracks, log=None):
    """accepted 0 -> f maps (3x3, small coords) per tracked frame, their agreement, the chain maps, frame size, the tracked
    frames and whether the camera is still; log (a list) receives one diagnostic row per frame"""
    cap = cv2.VideoCapture(path); W, H = cap.get(3), cap.get(4)
    frames = sorted({f for fr, _ in tracks.values() for f in fr})
    imgs = {}; k = 0; last = max(frames) if frames else 0
    while k <= last:
        ok, im = cap.read()
        if not ok: break
        if k in frames or k == 0: imgs[k] = cv2.cvtColor(cv2.resize(im, SZ, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        k += 1
    cap.release()
    if 0 not in imgs: return dict(A={}, score={}, chain={}, W=W, H=H, frames=frames, still=False)
    bp = {(k, b): bandpass(g, b) for k, g in imgs.items() for b in (BAND, BAND_SEL)}; tex = {kb: textured(d, kb[1]) for kb, d in bp.items()}
    box_at = {n: dict(zip(fr, xy)) for n, (fr, xy) in tracks.items()}
    sx, sy = SZ[0] / W, SZ[1] / H; S, Si = np.diag([sx, sy, 1.0]), np.diag([1 / sx, 1 / sy, 1.0])

    def boxes(f): return [b for b in (box_at[n].get(f) for n in tracks) if b is not None and not np.isnan(b[0])]

    def points(f):   # where the map is used: corners of the tracked boxes at frame f (small coords; none tracked: the image centre)
        return np.array([[x * sx, y * sy, 1] for b in boxes(f) for x in (b[0], b[2]) for y in (b[1], b[3])] or [[SZ[0] / 2, SZ[1] / 2, 1]], float)

    def corr(t, f, M, band):   # local correlation of frame t with frame f under the t -> f map M, on t's textured background
        back = Si @ np.linalg.inv(M) @ S
        return local_corr(bp[t, band], warped(imgs[f], M, band), (mask_for(boxes(t) + [_map_box(b, back) for b in boxes(f)], W, H) > 0) & tex[t, band], band)

    def score(t, f, M):   # agreement at the coarse scale (acceptance)
        r = corr(t, f, M, BAND); return 0.0 if r is None else float((r >= LNCC).mean())

    def fit(t, f, s, msk, fits):   # ECC from start s, kept unless it failed or repeats a fit
        D = ecc(imgs[t], imgs[f], s, msk)
        if D is not None and not any(np.abs(D - c).max() < 1e-6 for c in fits): fits.append(D)

    def pick(t, f, cands):   # the best mean local correlation at the fine scale, and its agreement; (None, 0) for no candidate
        if not cands: return None, 0.0
        sc = [-1.0 if r is None else float(r.mean()) for r in (corr(t, f, M, BAND_SEL) for M in cands)]; M = cands[int(np.argmax(sc))]
        return M, score(t, f, M)

    A = {0: I3}; sc_of = {0: 1.0}; chain = {0: I3}; chain_ok = {0: True}; direct = {0: I3}
    p = q = 0
    for f in frames:
        if f == 0 or f not in imgs: continue
        P = points(f); bf = boxes(f)
        nf = []; fit(q, f, I3, mask_for(boxes(q) + bf, W, H), nf); N, sn = pick(q, f, nf + [I3])
        pr = N @ chain[q]; tri = N @ direct[q] if q in direct else None
        b0 = [_map_box(b, Si @ pr @ S) for b in boxes(0)]   # frame-0 boxes masked where the chain puts them in frame f
        msk = mask_for(b0 + bf, W, H)

        def decide(fits):
            cands = fits + [I3]
            D, sd = pick(0, f, [M for M in cands if gap(M, pr, P) <= TOL])   # agrees with the chain
            if D is not None and sd >= S_MIN: return D, sd, True, 'chain'
            # after rejected frames the chain may have drifted: a map that agrees with q's direct map carried by the neighbour map,
            # lies within a few px per frame since p of the chain and explains frame 0 at least as well as the chain (an alias that
            # jumps far from the chain, e.g. along horizontal edges under a pan, or the identity kept under a pan, cannot pass)
            near = min(TRI_BASE + TRI_RATE * (f - p), TRI_MAX)
            D, sd = pick(0, f, [M for M in cands if tri is not None and gap(M, tri, P) <= TOL and gap(M, pr, P) <= near])
            if D is not None and sd >= S_MIN and sd >= score(0, f, pr) - 0.02: return D, sd, True, 'tri'
            D, sd = pick(0, f, cands); return D, sd, False, ''
        df = []; fit(0, f, pr, msk, df); D, sd, ok, how = decide(df)
        if not ok:   # other starts only when the chain's fit does not do
            for s in (A[p], I3):
                if np.abs(s - pr).max() > 1e-6: fit(0, f, s, msk, df)
            D, sd, ok, how = decide(df)
        direct[f] = D
        if ok: A[f] = D; sc_of[f] = sd; chain[f] = D; chain_ok[f] = True; p = f
        else: chain[f] = pr; chain_ok[f] = sn >= S_MIN and chain_ok[q]
        if log is not None: log.append(dict(f=f, sn=round(sn, 3), sd=round(sd, 3), si=round(score(0, f, I3), 3), g_chain=round(gap(D, pr, P), 2),
                                            g_tri=round(gap(D, tri, P), 2) if tri is not None else None, n_fit=len(df), ok=ok, how=how))
        q = f

    def moved(M, f):   # largest displacement (video px) that M gives to the frame-f points
        P = points(f); Q = P @ np.linalg.inv(M).T
        return float(np.hypot((Q[:, 0] - P[:, 0]) / sx, (Q[:, 1] - P[:, 1]) / sy).max())
    still = all(moved(M, f) < STILL_PX for f, M in A.items())
    if still:
        for f in frames:
            if f in imgs and f not in A and chain_ok.get(f) and moved(chain[f], f) < STILL_PX: A[f] = I3; sc_of[f] = 0.0
    return dict(A=A, score=sc_of, chain=chain, W=W, H=H, frames=frames, still=still)


def full_res(M, W, H):
    """0 -> f map (small coords) -> 2x3 matrix mapping frame-f pixels into frame-0 pixels (full resolution)"""
    S = np.diag([SZ[0] / W, SZ[1] / H, 1.0])
    return (np.linalg.inv(S) @ np.linalg.inv(M) @ S)[:2]


def frame_mats(r):
    """f -> 0 full-resolution 2x3 matrices of the accepted frames (identity for a still camera)"""
    return {f: (np.eye(2, 3) if r['still'] else full_res(M, r['W'], r['H'])) for f, M in r['A'].items()}


def camera_mats(path, tracks):
    """per tracked frame with an accepted map: 2x3 matrix mapping frame-f pixels into frame-0 pixels, agreement score, frame
    size, tracked frames"""
    r = register(path, tracks)
    return frame_mats(r), r['score'], r['W'], r['H'], r['frames']


def compensate(args):
    key, path, tracks = args          # tracks: {name: (frames list, xyxy list)}
    cv2.setNumThreads(1)
    r = register(path, tracks); W, H = r['W'], r['H']; mats = frame_mats(r)
    out = {n: (fr, [None if b is None or f not in mats else _map_box(b, np.vstack([mats[f], [0, 0, 1]])) for f, b in zip(fr, xy)])
           for n, (fr, xy) in tracks.items()}
    shifts, corners = [], []
    P = np.array([[W / 2, 0, 0, W, W], [H / 2, 0, H, 0, H], [1, 1, 1, 1, 1]], float)   # image centre and corners
    for M in r['A'].values():   # accepted frames only (a still camera keeps its measured value)
        c = full_res(M, W, H) @ P; d = np.hypot(c[0] - P[0], c[1] - P[1])
        shifts.append(float(d[0])); corners.append(float(d.max()))
    # F120 (2026-11-11): max_shift_px, the shift of the image centre, misses a zoom or dolly towards the centre (Wan2.2-I2V pushes in
    # to a close-up while the centre stays put); max_corner_px, the largest displacement of the image corners, measures it
    stats = dict(max_shift_px=round(max(shifts), 1) if shifts else 0.0, max_corner_px=round(max(corners), 1) if corners else 0.0,
                 frames_failed=sum(1 for f in r['frames'] if f not in mats), frames=len(r['frames']), still=bool(r['still']), size=[W, H])
    return key, out, stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('items'); ap.add_argument('tracks', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=9); ap.add_argument('--keys')
    a = ap.parse_args()
    items = {x['key']: x for x in json.load(open(a.items))['items'] if x.get('map')}
    keys = set(open(a.keys).read().split()) if a.keys else None
    T = collections.defaultdict(dict); meta = {}
    for p in a.tracks:
        for l in open(p):
            r = json.loads(l); v, n = r['id'].split('__', 1)
            if v not in items or (keys and v not in keys): continue
            T[v][n] = (r['tracks'][n]['frames'], r['tracks'][n]['xyxy']); meta[v] = r.get('prompt_frame', 0)
    import common as c2
    jobs = [(v, c2.gt(items[v]['path']), T[v]) for v in T]
    with get_context('fork').Pool(a.workers) as pool, open(a.out, 'w') as f, open(a.out.replace('.jsonl', '_stats.jsonl'), 'w') as fs:
        for v, out, stats in pool.imap_unordered(compensate, jobs, chunksize=4):
            for n, (fr, xy) in out.items():
                f.write(json.dumps(dict(id=f'{v}__{n}', prompt_frame=meta[v], tracks={n: dict(frames=fr, xyxy=xy)}, mode='lean+camcomp')) + '\n')
            fs.write(json.dumps(dict(key=v, **stats)) + '\n')
    print('compensated', len(jobs), 'videos ->', a.out)
