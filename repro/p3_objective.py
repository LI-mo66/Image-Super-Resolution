"""P3-only normalized RGB MSE objective; no model or inference change."""


def normalized_mse(sr, hr):
    if sr.shape != hr.shape:
        raise ValueError('P3 requires spatially aligned SR and HR')
    return ((sr - hr) / 255).square().mean()
