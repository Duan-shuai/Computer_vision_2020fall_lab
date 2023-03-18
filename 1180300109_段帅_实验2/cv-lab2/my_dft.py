import numpy as np


def shift_ft(img):
    M, N = img.shape
    shift = np.matrix([[pow(-1, i + j) for j in range(N)] for i in range(M)])
    U = np.matrix([[np.exp(-1j * 2 * 3.14159 * m * i / M) for m in range(M)] for i in range(M)])
    V = np.matrix([[np.exp(-1j * 2 * 3.14159 * n * j / N) for j in range(N)] for n in range(N)])
    return U.dot(np.multiply(img, shift)).dot(V)


def shift_ift(img):
    M, N = img.shape
    shift = np.matrix([[pow(-1, i + j) for j in range(N)] for i in range(M)])
    U = np.matrix([[np.exp(1j * 2 * 3.14159 * m * i / M) for m in range(M)] for i in range(M)])
    V = np.matrix([[np.exp(1j * 2 * 3.14159 * n * j / N) for j in range(N)] for n in range(N)])
    return np.multiply(shift, U.dot(img).dot(V)) / M / N


def dft(img):
    H, W = img.shape

    # Prepare DFT coefficient
    G = np.zeros((H, W), dtype=np.complex)
    # prepare processed index corresponding to original image positions
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    # dft
    for v in range(H):
        for u in range(W):
            G[v, u] = np.sum(img[...] * np.exp(-2j * np.pi * (x * u / W + y * v / H))) / np.sqrt(H * W)

    return G


def idft(G):
    H, W = G.shape
    out = np.zeros((H, W, channel), dtype=np.float32)

    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)


    for v in range(H):
        for u in range(W):
            out[v, u, c] = np.abs(np.sum(G[...] * np.exp(2j * np.pi * (x * u / W + y * v / H)))) / np.sqrt(W * H)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out