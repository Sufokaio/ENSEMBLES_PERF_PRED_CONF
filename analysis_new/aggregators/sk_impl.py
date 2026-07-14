# Scott-Knott clustering for ranking treatments.

import math
import numpy as np

def a12(lst1, lst2):
    x, y = np.asarray(lst1, float), np.asarray(lst2, float)
    more = (x[:, None] > y[None, :]).sum()
    same = (x[:, None] == y[None, :]).sum()
    return (more + 0.5 * same) / (len(x) * len(y))

def bootstrap(lst1, lst2, conf=0.01, b=1000, seed=42):
    rng = np.random.default_rng(seed)
    y, z = np.asarray(lst1, float), np.asarray(lst2, float)
    x = np.concatenate([y, z])
    y_mu, z_mu, x_mu = y.mean(), z.mean(), x.mean()
    s1, s2 = y.std(ddof=1), z.std(ddof=1)
    denom = math.sqrt(s1**2 / len(y) + s2**2 / len(z)) if s1 + s2 else 1.0
    tobs = (z_mu - y_mu) / denom if s1 + s2 else z_mu - y_mu

    yhat = y - y_mu + x_mu
    zhat = z - z_mu + x_mu

    yb = rng.choice(yhat, size=(b, len(yhat)), replace=True)
    zb = rng.choice(zhat, size=(b, len(zhat)), replace=True)
    yb_mu = yb.mean(axis=1)
    zb_mu = zb.mean(axis=1)
    s1b = yb.std(axis=1, ddof=1)
    s2b = zb.std(axis=1, ddof=1)
    d = np.sqrt(s1b**2 / len(y) + s2b**2 / len(z))
    d = np.where(d == 0, 1.0, d)
    t_boot = (zb_mu - yb_mu) / d
    return (t_boot > tobs).sum() / b < conf

def scott_knott(groups, cohen=0.3, small=3, use_a12=True, conf=0.01,
                a12_threshold=0.60, seed=42):
    def median(lst):
        lst = sorted(lst)
        n = len(lst)
        mid = n // 2
        return lst[mid] if n % 2 else (lst[mid - 1] + lst[mid]) / 2

    def iqr(lst):
        lst = sorted(lst)
        n = len(lst)
        return lst[int(n * 0.75)] - lst[int(n * 0.25)]

    def can_split(left, right):
        if len(left) < small or len(right) < small:
            return False
        if use_a12:
            if a12(right, left) < a12_threshold:
                return False
            if not bootstrap(left, right, conf=conf, seed=seed):
                return False
        else:
            if abs(median(left) - median(right)) <= cohen * (iqr(left + right) or 1):
                return False
        return True

    def recursive(groups, rank=1, ranks=None):
        if ranks is None:
            ranks = []
        if len(groups) == 1:
            ranks.append((rank, groups[0][0], median(groups[0][1]),
                          iqr(groups[0][1]), groups[0][1]))
            return ranks
        groups = sorted(groups, key=lambda g: median(g[1]))
        all_values = [v for g in groups for v in g[1]]
        best_score, best_split = 0, None
        for i in range(1, len(groups)):
            left  = [v for g in groups[:i] for v in g[1]]
            right = [v for g in groups[i:]  for v in g[1]]
            if not can_split(left, right):
                continue
            n  = len(all_values)
            mu = sum(all_values) / n
            score = (
                (len(left)  / n) * (mu - sum(left)  / len(left))  ** 2 +
                (len(right) / n) * (mu - sum(right) / len(right)) ** 2
            )
            if score > best_score:
                best_score, best_split = score, i
        if best_split:
            recursive(groups[:best_split], rank, ranks)
            next_rank = max(r[0] for r in ranks) + 1
            recursive(groups[best_split:], next_rank, ranks)
        else:
            for g in groups:
                ranks.append((rank, g[0], median(g[1]), iqr(g[1]), g[1]))
        return ranks

    return sorted(recursive(groups), key=lambda x: x[0])
