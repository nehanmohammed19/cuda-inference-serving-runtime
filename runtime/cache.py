"""Host page ownership, device-resident KV values. One pool per model layer.

Callers must synchronize outstanding device work before fork/release/reuse.
The synchronous engine enforces that boundary; this allocator is not thread safe.
"""
import numpy as np


class PagedCache:
    def __init__(self, xp, pages, page_size, heads, dim):
        if min(pages, page_size, heads, dim) <= 0:
            raise ValueError('cache dimensions must be positive')
        self.xp, self.page_size = xp, page_size
        self.k = xp.zeros((pages, page_size, heads, dim), dtype=xp.float32)
        self.v = xp.zeros_like(self.k)
        self.refs = np.zeros(pages, dtype=np.int32)
        self.free = list(reversed(range(pages)))
        self.tables, self.lengths = {}, {}
        self.peak_pages = 0

    def create(self, key):
        if key in self.tables:
            raise ValueError('duplicate sequence')
        self.tables[key], self.lengths[key] = [], 0

    def _allocate(self):
        if not self.free:
            raise MemoryError('KV page pool exhausted')
        page = self.free.pop()
        self.refs[page] = 1
        self.peak_pages = max(self.peak_pages, len(self.refs) - len(self.free))
        return page

    def fork(self, source, target):
        if target in self.tables:
            raise ValueError('duplicate sequence')
        self.tables[target] = list(self.tables[source])
        self.lengths[target] = self.lengths[source]
        for page in self.tables[target]:
            self.refs[page] += 1

    def append(self, key, k, v):
        position = self.lengths[key]
        table = self.tables[key]
        offset = position % self.page_size
        if offset == 0:
            table.append(self._allocate())
        elif self.refs[table[-1]] > 1:
            # A fork may share a partially filled page. Never overwrite its tail.
            old = table[-1]
            new = self._allocate()  # allocation failure leaves ownership unchanged
            self.k[new] = self.k[old]
            self.v[new] = self.v[old]
            self.refs[old] -= 1
            table[-1] = new
        self.k[table[-1], offset] = k
        self.v[table[-1], offset] = v
        self.lengths[key] += 1

    def release(self, key):
        for page in self.tables.pop(key):
            self.refs[page] -= 1
            if self.refs[page] == 0:
                self.free.append(page)
        del self.lengths[key]

    def dense(self, key):
        table, length = self.tables[key], self.lengths[key]
        shape = (-1,) + self.k.shape[2:]
        return (self.k[table].reshape(shape)[:length],
                self.v[table].reshape(shape)[:length])

    def metadata(self, keys):
        width = max(len(self.tables[key]) for key in keys)
        tables = np.full((len(keys), width), -1, dtype=np.int32)
        for row, key in enumerate(keys):
            tables[row, :len(self.tables[key])] = self.tables[key]
        lengths = np.array([self.lengths[key] for key in keys], dtype=np.int32)
        return self.xp.asarray(tables), self.xp.asarray(lengths)

    def check(self):
        expected = np.zeros_like(self.refs)
        for table in self.tables.values():
            for page in table:
                expected[page] += 1
        assert np.array_equal(expected, self.refs)
        assert set(self.free) == set(np.flatnonzero(self.refs == 0))
        assert len(self.free) == len(set(self.free))
