from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

@dataclass
class Page:
    page_id: int
    seq_id: int | None = None
    start_pos: int = 0
    filled: int = 0
    valid: bool = False

class PagedKVCache:
    """Simulates vLLM-style PagedAttention KV cache.

    CPU/GPU naive tensor storage with page-level allocation. Works
    deterministically for small-scale inference and unit tests.
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 32,
        max_pages: int = 1024,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype
        self.device = device

        self.free_pages: list[int] = list(range(max_pages))
        self.seq_pages: dict[int, list[Page]] = {}
        self.seq_last_access: dict[int, int] = {}
        self._seq_batch_size: dict[int, int] = {}
        self._access_counter = 0

        self._pages = torch.zeros(
            max_pages, num_layers, 2, num_heads, page_size, head_dim,
            dtype=dtype, device=device,
        )
        self._pages_meta: list[Page] = [Page(page_id=i) for i in range(max_pages)]

    def _alloc_page(self, seq_id: int) -> Page:
        if self.free_pages:
            pid = self.free_pages.pop()
        else:
            pid = self._evict_oldest()
        page = self._pages_meta[pid]
        page.seq_id = seq_id
        page.filled = 0
        page.valid = True
        self.seq_pages.setdefault(seq_id, []).append(page)
        return page

    def _evict_oldest(self) -> int:
        if not self.seq_last_access:
            raise RuntimeError("PagedKVCache: no pages to evict, set larger max_pages")
        oldest_seq = min(self.seq_last_access, key=self.seq_last_access.get)
        pages = self.seq_pages.pop(oldest_seq, [])
        for p in pages:
            p.valid = False
            p.seq_id = None
            p.filled = 0
            self.free_pages.append(p.page_id)
        self.seq_last_access.pop(oldest_seq, None)
        return self.free_pages.pop()

    def num_active_seqs(self) -> int:
        return len(self.seq_pages)

    def touch(self, seq_id: int) -> None:
        self._access_counter += 1
        self.seq_last_access[seq_id] = self._access_counter

    def put(
        self,
        seq_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int = 0,
    ) -> int:
        self.touch(seq_id)
        seq_len = k.shape[-2]
        if seq_id not in self._seq_batch_size:
            self._seq_batch_size[seq_id] = k.shape[0]
        total_written = 0
        cur = start_pos
        end = start_pos + seq_len
        pages = self.seq_pages.setdefault(seq_id, [])
        while cur < end:
            page_idx = cur // self.page_size
            offset_in_page = cur % self.page_size
            while len(pages) <= page_idx:
                self._alloc_page(seq_id)
                pages = self.seq_pages[seq_id]
            page = pages[page_idx]
            slots_left = self.page_size - offset_in_page
            tokens_now = min(slots_left, end - cur)
            src_start = cur - start_pos
            src_end = src_start + tokens_now
            k_slice = k[:, :, src_start:src_end, :]
            v_slice = v[:, :, src_start:src_end, :]
            num_layers = k_slice.shape[0] if k_slice.shape[0] == self.num_layers else 1
            for li in range(num_layers):
                k_tok = k_slice[li] if num_layers > 1 else k_slice[0]
                v_tok = v_slice[li] if num_layers > 1 else v_slice[0]
                self._pages[page.page_id, li, 0, :, offset_in_page:offset_in_page+tokens_now, :] = k_tok
                self._pages[page.page_id, li, 1, :, offset_in_page:offset_in_page+tokens_now, :] = v_tok
            page.filled = max(page.filled, offset_in_page + tokens_now)
            cur += tokens_now
            total_written += tokens_now
        return total_written

    def get(self, seq_id: int, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.touch(seq_id)
        pages = self.seq_pages.get(seq_id, [])
        bs = self._seq_batch_size.get(seq_id, 1)
        if not pages:
            empty = torch.zeros(bs, self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device)
            mask = torch.zeros(bs, 0, dtype=self.dtype, device=self.device)
            return empty, empty, mask
        total = len(pages) * self.page_size
        K = torch.zeros(bs, self.num_heads, total, self.head_dim, dtype=self.dtype, device=self.device)
        V = torch.zeros(bs, self.num_heads, total, self.head_dim, dtype=self.dtype, device=self.device)
        mask = torch.zeros(bs, total, dtype=self.dtype, device=self.device)
        for b in range(bs):
            li = b if (bs == self.num_layers and b < self.num_layers) else layer_idx
            for i, p in enumerate(pages):
                base = i * self.page_size
                K[b, :, base:base+self.page_size, :] = self._pages[p.page_id, li, 0, :, :, :]
                V[b, :, base:base+self.page_size, :] = self._pages[p.page_id, li, 1, :, :, :]
                filled = min(p.filled, self.page_size)
                if filled > 0:
                    mask[b, base:base+filled] = 1.0
        return K, V, mask

    def clear(self, seq_id: int) -> None:
        pages = self.seq_pages.pop(seq_id, [])
        for p in pages:
            p.seq_id = None
            p.filled = 0
            p.valid = False
            self.free_pages.append(p.page_id)
        self.seq_last_access.pop(seq_id, None)
        self._seq_batch_size.pop(seq_id, None)
