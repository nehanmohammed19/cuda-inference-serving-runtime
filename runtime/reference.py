"""Dense numerical oracle; deliberately independent of paged CUDA indexing."""
import numpy as np


def rmsnorm(x, weight, eps=1e-5):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * weight


def rope(x, positions, theta=10000.0):
    d = x.shape[-1]
    if d % 2: raise ValueError('RoPE needs an even head dimension')
    angle = np.asarray(positions)[:, None] * theta ** (-np.arange(d // 2) * 2.0 / d)
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
    a, b = x[..., :d // 2], x[..., d // 2:]
    return np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1).astype(np.float32)


def attention(q, k, v):
    # q: Hq,D; k/v: T,Hkv,D. Consecutive query heads share one KV head.
    if q.shape[0] % k.shape[1]: raise ValueError('invalid GQA ratio')
    groups = q.shape[0] // k.shape[1]
    indices = np.arange(q.shape[0]) // groups
    scores = np.einsum('hd,thd->ht', q, k[:, indices]) / np.sqrt(q.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    probability = np.exp(scores)
    probability /= probability.sum(axis=-1, keepdims=True)
    return np.einsum('ht,thd->hd', probability, v[:, indices]).astype(np.float32)


def dense_causal(q, k, v):
    return np.stack([attention(q[t], k[:t+1], v[:t+1]) for t in range(len(q))])
