"""Run: python k8s_experiments.py   (needs data/k8s_raw.jsonl, gda.py, k8s_pipeline.py)
Grid: target definition x min year x +sig one-hot, with 3 models incl. a TF-IDF LogReg 'text ceiling'."""
import json, itertools
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from k8s_pipeline import RAW, NUM, LOG, build_row
from gda import GDA

def load(target: str, min_year: int) -> pd.DataFrame:
    rows = []
    for line in RAW.open(encoding="utf-8"):
        it = json.loads(line); r = build_row(it)
        if r is None: continue
        names = [l["name"] for l in it["labels"]]
        r["sigs"] = [n for n in names if n.startswith("sig/")]
        if target == "urgent":                       # only critical-urgent counts as escalation
            r["y"] = int("priority/critical-urgent" in names)
        rows.append(r)
    df = pd.DataFrame(rows)
    df = df[pd.to_datetime(df.created_at).dt.year >= min_year]
    return df.sort_values("created_at").reset_index(drop=True)

def pick_thr(y, p, min_recall=0.80):
    grid = np.linspace(0.02, 0.98, 97)
    ok = [t for t in grid if recall_score(y, p >= t, zero_division=0) >= min_recall]
    return max(ok, key=lambda t: precision_score(y, p >= t, zero_division=0)) if ok else 0.5

def run(df, use_sig, svd_dims=20):
    n = len(df); a, b = int(.70 * n), int(.85 * n)
    tr, va, te = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    top = pd.Series([s for l in tr.sigs for s in l]).value_counts().head(15).index.tolist() if use_sig else []

    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, stop_words="english", sublinear_tf=True)
    Str = tf.fit_transform(tr.text); svd = TruncatedSVD(svd_dims, random_state=0).fit(Str)

    def dense(d, S):
        X = d[NUM].astype(float).copy(); X[LOG] = np.log1p(X[LOG])
        sig = np.array([[s in l for s in top] for l in d.sigs], float).reshape(len(d), len(top))
        return np.hstack([X.to_numpy(), sig, svd.transform(S)])
    Xtr = dense(tr, Str); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Sva, Ste = tf.transform(va.text), tf.transform(te.text)
    Xva, Xte = (dense(va, Sva) - mu) / sd, (dense(te, Ste) - mu) / sd

    # text ceiling: full sparse TF-IDF + scaled numerics into LogReg (no SVD bottleneck)
    sp = lambda X, S: hstack([S, csr_matrix(X[:, :len(NUM) + len(top)])]).tocsr()
    sets = {"GDA-LDA": (GDA(shared_cov=True, reg=1e-3), Xtr, Xva, Xte),
            "LogReg": (LogisticRegression(max_iter=2000), Xtr, Xva, Xte),
            "TFIDF-LogReg": (LogisticRegression(max_iter=3000, C=2.0), sp(Xtr, Str), sp(Xva, Sva), sp(Xte, Ste))}
    out = []
    for name, (m, A, V, T) in sets.items():
        m.fit(A, tr.y.to_numpy())
        pv, pt = m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]
        t = pick_thr(va.y, pv)
        out.append({"model": name, "base": te.y.mean(), "auc": roc_auc_score(te.y, pt),
                    "pr_auc": average_precision_score(te.y, pt),
                    "prec@R>=.8": precision_score(te.y, pt >= t, zero_division=0),
                    "recall": recall_score(te.y, pt >= t, zero_division=0), "n": n})
    return out

if __name__ == "__main__":
    res = []
    for target, yr, sig in itertools.product(["soon_or_urgent", "urgent"], [2015, 2019], [False, True]):
        df = load(target, yr)
        for r in run(df, sig):
            res.append({"target": target, "from": yr, "sig": sig, **r})
        print("done", target, yr, sig, flush=True)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(res).round(3).to_string(index=False))