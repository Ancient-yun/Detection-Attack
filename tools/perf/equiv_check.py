"""Equivalence harness for attack tensor-optimization work.

Runs a decision-based attack deterministically on a fixed image/seed and
produces a *signature* capturing everything that must stay identical:
  - final adversarial image (exact float32 bytes + sha256)
  - total query count
  - final L0 distance
  - full L0 trace
  - original / adversarial detections (bboxes, labels, scores)

Two modes:
  run     : execute an attack and dump a signature (.npz)
  compare : diff two signatures at 'byte' (Option A) or 'metric' (Option B) level

Determinism: every global RNG (random, numpy, torch) is reseeded before the
attack, and cudnn is put in deterministic mode. The first thing to validate is
that running the *same* code twice yields a byte-identical signature.
"""
import argparse
import hashlib
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace")
from adversarial_attack.attack_pipeline import DetectionAttackPipeline  # noqa: E402

MODELS = {
    "atss": (
        "configs/atss/atss_r50_fpn_1x_coco.py",
        "ckpt/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth",
    ),
}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _detsig(d):
    return {
        "bboxes": np.asarray(d["bboxes"], dtype=np.float32),
        "labels": np.asarray(d["labels"]).astype(np.int64),
        "scores": np.asarray(d["scores"], dtype=np.float32),
    }


def run(args):
    cfg, ckpt = MODELS[args.model]
    _seed_all(args.seed)
    pipe = DetectionAttackPipeline(
        model_type="mmdet",
        config_path=cfg,
        checkpoint_path=ckpt,
        attack_method=args.attack,
        device="cuda:0",
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        success_thr=args.success_thr,
        seed=args.seed if args.attack == "sparse_evo" else None,
        **({"pop_size": 10, "cr": 0.9, "mu": 0.01}
           if args.attack == "sparse_evo" else {}),
    )
    pipe.verbose = False
    if hasattr(pipe.attack, "verbose"):
        pipe.attack.verbose = False

    oimg = pipe.load_image(args.image)
    orig_dets = pipe.model.set_reference(oimg)

    # Reseed right before the stochastic part so the starting point + attack
    # RNG stream is fully controlled and independent of model-load order.
    _seed_all(args.seed)
    start, sq = pipe.generate_starting_point(oimg, 0, args.seed)

    remaining = args.max_query - sq
    if args.attack == "sparse_evo":
        adv, aq, trace, _ = pipe.attack.evo_perturb(
            oimg, start, 0, -1, max_query=remaining, snapshot_interval=0)
        adv_img = adv
        l0_trace = trace.detach().cpu().numpy().astype(np.int64)
    else:  # pointwise_multi_sched
        oimg_np = oimg.cpu().numpy()
        timg_np = start.cpu().numpy()
        npix = 0.1
        adv_flat, aq, trace, _ = pipe.attack.pw_perturb_multiple_scheduling(
            oimg_np, timg_np, 0, -1, npix=npix,
            max_query=remaining, snapshot_interval=0)
        adv_img = torch.from_numpy(
            adv_flat.reshape(oimg.shape)).float().to(oimg.device)
        l0_trace = np.asarray(trace, dtype=np.int64)

    adv_dets = pipe.model.predict(adv_img)

    adv_np = adv_img.detach().cpu().numpy().astype(np.float32)
    sha = hashlib.sha256(adv_np.tobytes()).hexdigest()
    total_q = sq + aq

    o = _detsig(orig_dets)
    a = _detsig(adv_dets)
    np.savez(
        args.out,
        adv=adv_np,
        adv_sha=sha,
        total_queries=np.int64(total_q),
        l0_trace=l0_trace,
        final_l0=np.int64(int(l0_trace[-1]) if len(l0_trace) else 0),
        o_bboxes=o["bboxes"], o_labels=o["labels"], o_scores=o["scores"],
        a_bboxes=a["bboxes"], a_labels=a["labels"], a_scores=a["scores"],
    )
    print(f"[run] attack={args.attack} seed={args.seed} "
          f"total_q={total_q} final_l0={int(l0_trace[-1]) if len(l0_trace) else 0} "
          f"adv_sha={sha[:16]} -> {args.out}")


def compare(args):
    r = np.load(args.ref)
    t = np.load(args.test)
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] {name} {detail}")

    if args.mode == "byte":
        same_sha = str(r["adv_sha"]) == str(t["adv_sha"])
        line("adv image sha256", same_sha)
        if not same_sha:
            diff = np.abs(r["adv"].astype(np.float64) - t["adv"].astype(np.float64))
            line("  adv exact equal", bool((diff == 0).all()),
                 f"max|Δ|={diff.max():.3e} n_diff={(diff>0).sum()}")
        line("total_queries", int(r["total_queries"]) == int(t["total_queries"]),
             f"{int(r['total_queries'])} vs {int(t['total_queries'])}")
        line("l0_trace exact",
             r["l0_trace"].shape == t["l0_trace"].shape
             and bool((r["l0_trace"] == t["l0_trace"]).all()))
        for k in ("a_bboxes", "a_labels", "a_scores"):
            g = r[k].shape == t[k].shape and bool(np.array_equal(r[k], t[k]))
            line(f"adv_{k}", g)
    else:  # metric
        line("total_queries (±0)",
             int(r["total_queries"]) == int(t["total_queries"]),
             f"{int(r['total_queries'])} vs {int(t['total_queries'])}")
        rl, tl = int(r["final_l0"]), int(t["final_l0"])
        rel = abs(rl - tl) / max(rl, 1)
        line(f"final_l0 (±{args.l0_tol:.0%})", rel <= args.l0_tol,
             f"{rl} vs {tl} (rel {rel:.2%})")
        line("adv detection count",
             len(r["a_labels"]) == len(t["a_labels"]),
             f"{len(r['a_labels'])} vs {len(t['a_labels'])}")

    print(f"\n==> {'PASS' if ok else 'MISMATCH'} (mode={args.mode})")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--attack", required=True,
                    choices=["sparse_evo", "pointwise_multi_sched"])
    pr.add_argument("--image", required=True)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--model", default="atss", choices=list(MODELS))
    pr.add_argument("--max-query", type=int, default=200)
    pr.add_argument("--score-thr", type=float, default=0.5)
    pr.add_argument("--iou-thr", type=float, default=0.5)
    pr.add_argument("--success-thr", type=float, default=0.7)
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=run)

    pc = sub.add_parser("compare")
    pc.add_argument("--ref", required=True)
    pc.add_argument("--test", required=True)
    pc.add_argument("--mode", default="byte", choices=["byte", "metric"])
    pc.add_argument("--l0-tol", type=float, default=0.02)
    pc.set_defaults(func=compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
