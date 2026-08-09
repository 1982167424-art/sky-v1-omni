from __future__ import annotations
import torch
import torch.nn as nn
from .config import SkyModelConfig, ModalConfig, HeadsConfig, build_model_from_config
from .backbone import UniTransformerBackbone
from .embeddings import ModalTypeEmbedding
from .modal_tokenizers import (
    TextTokenizer, ImageTokenizer, AudioTokenizer, VideoTokenizer, ThreeDTokenizer,
)
from .modal_heads import (
    TextHead, ImageHead, AudioHead, VideoHead, ThreeDHead,
)

def _default_modal_cfg(name: str, mid: int) -> ModalConfig:
    return ModalConfig(modal_id=mid)

def _default_head_cfg() -> HeadsConfig:
    return HeadsConfig()

class SkyModel(nn.Module):
    def __init__(self, config: SkyModelConfig):
        super().__init__()
        self.config = config
        self.backbone = UniTransformerBackbone(config)
        self.modal_type_emb = ModalTypeEmbedding(num_modal_types=5, hidden_size=config.hidden_size)
        self.modal_types = config.modal_types
        modal = config.modal
        heads = config.heads
        text_m = modal.get("text", _default_modal_cfg("text", 0))
        self.text_tok = TextTokenizer(
            vocab_size=text_m.vocab_size,
            hidden_size=config.hidden_size,
            modal_id=text_m.modal_id,
        )
        self.image_tok = ImageTokenizer(modal.get("image", _default_modal_cfg("image", 1)), config.hidden_size)
        self.audio_tok = AudioTokenizer(modal.get("audio", _default_modal_cfg("audio", 2)), config.hidden_size)
        self.video_tok = VideoTokenizer(modal.get("video", _default_modal_cfg("video", 3)), config.hidden_size)
        self.threed_tok = ThreeDTokenizer(modal.get("three_d", _default_modal_cfg("three_d", 4)), config.hidden_size)
        self.text_head = TextHead(heads.get("text", _default_head_cfg()), config.hidden_size)
        img_m = modal.get("image", _default_modal_cfg("image", 1))
        self.image_head = ImageHead(heads.get("image", _default_head_cfg()), config.hidden_size, image_size=img_m.image_size)
        self.audio_head = AudioHead(heads.get("audio", _default_head_cfg()), config.hidden_size)
        vid_m = modal.get("video", _default_modal_cfg("video", 3))
        self.video_head = VideoHead(heads.get("video", _default_head_cfg()), config.hidden_size, frame_size=vid_m.frame_size)
        self.threed_head = ThreeDHead(heads.get("three_d", _default_head_cfg()), config.hidden_size)
        self._seg: list[tuple[int,int]] = []

    def _encode(self, inputs: dict) -> tuple[torch.Tensor, list[tuple[str, int]]]:
        device = next(self.parameters()).device
        segs: list[tuple[str, int]] = []
        embs_list: list[torch.Tensor] = []
        B = None
        if "text" in inputs and inputs["text"] is not None:
            t = inputs["text"].to(device)
            e = self.text_tok(t)
            e = e + self.modal_type_emb(torch.tensor(0, device=device)).view(1,1,-1)
            B = B or e.size(0)
            segs.append(("text", e.size(1)))
            embs_list.append(e)
        if "image" in inputs and inputs["image"] is not None:
            im = inputs["image"].to(device)
            e = self.image_tok(im)
            e = e + self.modal_type_emb(torch.tensor(1, device=device)).view(1,1,-1)
            B = B or e.size(0)
            segs.append(("image", e.size(1)))
            embs_list.append(e)
        if "audio" in inputs and inputs["audio"] is not None:
            a = inputs["audio"].to(device)
            e = self.audio_tok(a)
            e = e + self.modal_type_emb(torch.tensor(2, device=device)).view(1,1,-1)
            B = B or e.size(0)
            segs.append(("audio", e.size(1)))
            embs_list.append(e)
        if "video" in inputs and inputs["video"] is not None:
            v = inputs["video"].to(device)
            e = self.video_tok(v)
            e = e + self.modal_type_emb(torch.tensor(3, device=device)).view(1,1,-1)
            B = B or e.size(0)
            segs.append(("video", e.size(1)))
            embs_list.append(e)
        if "three_d" in inputs and inputs["three_d"] is not None:
            td = inputs["three_d"]
            if isinstance(td, (tuple, list)):
                pts = td[0].to(device)
                mv = td[1].to(device) if len(td) > 1 and td[1] is not None else None
                e = self.threed_tok(pts, mv)
            else:
                e = self.threed_tok(td.to(device))
            e = e + self.modal_type_emb(torch.tensor(4, device=device)).view(1,1,-1)
            B = B or e.size(0)
            segs.append(("three_d", e.size(1)))
            embs_list.append(e)
        if not embs_list:
            raise ValueError("SkyModel forward got empty inputs dict (no modalities provided)")
        embs = torch.cat(embs_list, dim=1)
        if embs.size(1) > self.config.max_position_embeddings:
            embs = embs[:, : self.config.max_position_embeddings]
            segs_trim: list[tuple[str, int]] = []
            used = 0
            for name, n in segs:
                take = max(0, min(n, self.config.max_position_embeddings - used))
                segs_trim.append((name, take))
                used += take
                if used >= self.config.max_position_embeddings:
                    break
            segs = segs_trim
        return embs, segs

    def forward(self, inputs: dict) -> dict:
        embs, segs = self._encode(inputs)
        B, S, H = embs.shape
        last = self.backbone(embs)
        cursor = 0
        seg_ranges: dict[str, tuple[int,int]] = {}
        for name, n in segs:
            seg_ranges[name] = (cursor, cursor + n)
            cursor += n
        def _seg(name: str) -> torch.Tensor:
            if name in seg_ranges:
                a, b = seg_ranges[name]
                return last[:, a:b] if b > a else last.mean(dim=1, keepdim=True)
            return last.mean(dim=1, keepdim=True)
        text_seg = _seg("text") if "text" in seg_ranges else last
        img_tokens_needed = self.image_head.nh * self.image_head.nh
        if "image" in seg_ranges:
            image_seg = _seg("image")
        else:
            image_seg = last.mean(dim=1, keepdim=True).repeat(1, img_tokens_needed, 1)
        image_out = self.image_head(image_seg)
        audio_seg = _seg("audio") if "audio" in seg_ranges else last
        video_seg = _seg("video") if "video" in seg_ranges else last
        threed_seg = _seg("three_d") if "three_d" in seg_ranges else last
        text_logits = self.text_head(text_seg)
        audio_out = self.audio_head(audio_seg)
        video_out = self.video_head(video_seg)
        three_d_out = self.threed_head(threed_seg)
        self._seg = []
        for k in ("text","image","audio","video","three_d"):
            if k in seg_ranges:
                self._seg.append(seg_ranges[k])
            else:
                self._seg.append((0, 0))
        return {
            "text": text_logits,
            "image": image_out,
            "audio": audio_out,
            "video": video_out,
            "three_d": three_d_out,
            "_segments": self._seg,
        }
