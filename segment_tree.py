import numpy as np


class SegmentTree:
    def __init__(self, array):
        self.values = np.asarray(array, dtype=float).ravel()
        self.n = len(self.values)
        if self.n == 0:
            raise ValueError("SegmentTree requires a non-empty array")
        self.tree = np.full(2 * self.n, -np.inf, dtype=float)
        self._build()

    def _build(self):
        self.tree[self.n:] = self.values
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])

    def range_max(self, l, r):
        if l < 0 or r >= self.n or l > r:
            raise IndexError(f"invalid interval [{l}, {r}] for size {self.n}")
        l += self.n
        r += self.n + 1
        best = -np.inf
        while l < r:
            if l & 1:
                best = max(best, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                best = max(best, self.tree[r])
            l >>= 1
            r >>= 1
        return best

    def range_max_index(self, l, r):
        if l < 0 or r >= self.n or l > r:
            raise IndexError(f"invalid interval [{l}, {r}] for size {self.n}")
        return int(l + np.argmax(self.values[l:r + 1]))

    def peak_above(self, l, r, threshold):
        return self.range_max(l, r) > threshold

    def update(self, index, value):
        if index < 0 or index >= self.n:
            raise IndexError(f"index {index} out of range for size {self.n}")
        i = index + self.n
        self.tree[i] = value
        self.values[index] = value
        i >>= 1
        while i >= 1:
            self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])
            i >>= 1


def peaks_from_spectrogram(spectrogram):
    return spectrogram.max(axis=0)



