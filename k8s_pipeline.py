"""Real-data pipeline: kubernetes/kubernetes issues -> escalation classifier (GDA).

Run:  export GITHUB_TOKEN=ghp_xxx   (needed: 60 req/h anonymous vs 5000/h with token)
      python k8s_pipeline.py fetch      # ~5 labels x N pages -> data/k8s_raw.jsonl
      python k8s_pipeline.py train      # features -> GDA vs LogReg -> metrics
Needs your GDA class saved as gda.py (copy notebook cell 4).
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = "kubernetes/kubernetes"
RAW = Path("data/k8s_raw.jsonl")

# label -> target. Only issues that carry one of these are used (i.e. triaged issues).
LABEL_TO_Y = {
    "priority/critical-urgent": 1,
    "priority/important-soon": 1,
    "priority/important-longterm": 0,
    "priority/backlog": 0,
}

# ------------------------------------------------------------------ fetch
def _get(url, params, headers):
    while True:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code in (403, 429):
            if r.headers.get("Retry-After"):                       # secondary rate limit
                wait = int(r.headers["Retry-After"]) + 2
            elif r.headers.get("X-RateLimit-Remaining") == "0":    # primary rate limit
                wait = max(int(r.headers.get("X-RateLimit-Reset", 0)) - time.time(), 0) + 5
            else:
                wait = 60
            print(f"rate limited, sleeping {wait:.0f}s"); time.sleep(wait); continue
        return r

def fetch(start_year: int = 2015, max_pages: int = 10):
    """Search API (is:issue -> no PRs), one query per label x quarter.
    Each query is capped at 1000 results by GitHub, so quarterly windows keep us under it."""
    h = {"Accept": "application/vnd.github+json"}
    if tok := os.getenv("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {tok}"
    RAW.parent.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with RAW.open("w", encoding="utf-8") as f:
        for label in LABEL_TO_Y:
            for per in pd.period_range(f"{start_year}Q1", pd.Timestamp.today(), freq="Q"):
                q = (f'repo:{REPO} is:issue label:"{label}" '
                     f'created:{per.start_time.date()}..{per.end_time.date()}')
                for page in range(1, max_pages + 1):
                    r = _get("https://api.github.com/search/issues",
                             {"q": q, "per_page": 100, "page": page}, h)
                    if r.status_code == 422:      # past the 1000-result cap -> move on
                        break
                    r.raise_for_status()
                    items = r.json().get("items", [])
                    for it in items:
                        if "pull_request" in it or it["number"] in seen:
                            continue
                        seen.add(it["number"]); n += 1
                        f.write(json.dumps(it) + "\n")
                    if len(items) < 100:
                        break
                print(f"{label} {per}: total {n}")
                time.sleep(2.2)                   # search API: 30 req/min

# ------------------------------------------------------------------ features
SEV_WORDS = ["panic", "crash", "data loss", "outage", "segfault", "oom", "deadlock",
             "regression", "cve", "security", "vulnerab", "corrupt", "unable to", "cannot", "broken"]
ASSOC = {"NONE": 0, "FIRST_TIME_CONTRIBUTOR": 0, "FIRST_TIMER": 0, "CONTRIBUTOR": 1,
         "COLLABORATOR": 2, "MEMBER": 2, "OWNER": 2}
SIG = re.compile(r"^sig/")

def build_row(it: dict) -> dict | None:
    names = [l["name"] for l in it["labels"]]
    ys = [LABEL_TO_Y[n] for n in names if n in LABEL_TO_Y]
    if not ys:
        return None
    y = max(ys)                                           # conflicting labels -> take more urgent
    title, body = it.get("title") or "", it.get("body") or ""
    text = f"{title}\n{body}".lower()
    return {
        "number": it["number"], "created_at": it["created_at"], "y": y,
        # --- creation-time features only (no comments/reactions: those are post-triage leakage) ---
        "title_len": len(title), "body_len": len(body),
        "code_blocks": body.count("```") // 2,
        "stack_markers": len(re.findall(r"goroutine \d+|panic:|traceback|\bat .+\.go:\d+", body, re.I)),
        "n_urls": len(re.findall(r"https?://", body)),
        "sev_hits": sum(w in text for w in SEV_WORDS),
        "assoc": ASSOC.get(it.get("author_association", "NONE"), 0),
        "n_sigs": sum(bool(SIG.match(n)) for n in names),
        "is_bug": int("kind/bug" in names),
        "text": f"{title}\n{body[:3000]}",
    }

def build_dataset() -> pd.DataFrame:
    rows = [build_row(json.loads(l)) for l in RAW.open(encoding="utf-8")]
    df = pd.DataFrame([r for r in rows if r]).sort_values("created_at").reset_index(drop=True)
    print(df.shape, "| escalate rate", f"{df.y.mean():.1%}")
    return df

# ------------------------------------------------------------------ model
NUM = ["title_len", "body_len", "code_blocks", "stack_markers", "n_urls", "sev_hits", "assoc", "n_sigs", "is_bug"]
LOG = ["title_len", "body_len", "code_blocks", "stack_markers", "n_urls"]

def train_eval(df: pd.DataFrame, svd_dims: int = 20, min_recall: float = 0.80):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
    from gda import GDA

    # Temporal split: train on oldest 70%, validate next 15%, test newest 15% (no future leakage).
    n = len(df); a, b = int(.70 * n), int(.85 * n)
    tr, va, te = df.iloc[:a], df.iloc[a:b], df.iloc[b:]

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, stop_words="english", sublinear_tf=True)
    svd = TruncatedSVD(svd_dims, random_state=0)
    Ttr = svd.fit_transform(tfidf.fit_transform(tr.text))

    def feats(d, T):
        X = d[NUM].astype(float).copy()
        X[LOG] = np.log1p(X[LOG])
        return np.hstack([X.to_numpy(), T])
    Xtr = feats(tr, Ttr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9          # train-only statistics
    prep = lambda d: (feats(d, svd.transform(tfidf.transform(d.text))) - mu) / sd
    Xtr, Xva, Xte = (Xtr - mu) / sd, prep(va), prep(te)

    models = {"GDA-LDA": GDA(shared_cov=True, reg=1e-3),
              "GDA-QDA": GDA(shared_cov=False, reg=1e-3),
              "LogReg": LogisticRegression(max_iter=2000, class_weight=None)}
    for name, m in models.items():
        m.fit(Xtr, tr.y.to_numpy())
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        # threshold: max precision s.t. recall >= min_recall on val
        grid = np.linspace(0.02, 0.98, 97)
        ok = [t for t in grid if recall_score(va.y, pv >= t, zero_division=0) >= min_recall]
        t = max(ok, key=lambda t: precision_score(va.y, pv >= t, zero_division=0)) if ok else 0.5
        print(f"{name:8s} AUC {roc_auc_score(te.y, pt):.3f} | PR-AUC {average_precision_score(te.y, pt):.3f} | "
              f"@thr {t:.2f}: P {precision_score(te.y, pt >= t, zero_division=0):.3f} "
              f"R {recall_score(te.y, pt >= t, zero_division=0):.3f}")
    print("baseline (always escalate) precision:", round(te.y.mean(), 3))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    fetch() if cmd == "fetch" else train_eval(build_dataset())