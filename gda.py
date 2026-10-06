import numpy as np


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class GDA:
    def __init__(self, shared_cov: bool = True, reg: float = 1e-6):
        self.shared_cov = shared_cov
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        K = len(self.classes_)

        self.priors_ = np.array([(y == k).mean() for k in self.classes_])      # phi
        self.means_ = np.array([X[y == k].mean(axis=0) for k in self.classes_])  # mu_k

        scatters = []
        for i, k in enumerate(self.classes_):
            Xc = X[y == k] - self.means_[i]
            scatters.append(Xc.T @ Xc)

        if self.shared_cov:
            pooled = sum(scatters) / n
            self.covs_ = np.array([pooled + self.reg * np.eye(d)] * K)
        else:
            counts = [(y == k).sum() for k in self.classes_]
            self.covs_ = np.array(
                [s / c + self.reg * np.eye(d) for s, c in zip(scatters, counts)]
            )

        self.inv_covs_ = np.array([np.linalg.inv(c) for c in self.covs_])
        self.logdets_ = np.array([np.linalg.slogdet(c)[1] for c in self.covs_])
        self.cond_ = float(np.linalg.cond(self.covs_[0]))  # diagnostic: singularity check
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        out = np.empty((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            diff = X - self.means_[i]
            maha = np.einsum("nd,de,ne->n", diff, self.inv_covs_[i], diff)
            out[:, i] = np.log(self.priors_[i]) - 0.5 * (
                d * np.log(2 * np.pi) + self.logdets_[i] + maha
            )
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        return np.exp(jll - _logsumexp(jll)[:, None])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary: threshold applies to P(class = classes_[1])."""
        if len(self.classes_) == 2:
            p = self.predict_proba(X)[:, 1]
            return np.where(p >= threshold, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]