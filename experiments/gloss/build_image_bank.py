"""Build a broad image-retrieval bank: one Qwen prototype per class over ALL
classes in the ImageNet root (merged 1k+21k). Used to retrieve gloss->image
anchors. 16 images/class to keep it fast.
"""
import numpy as np, torch, pathlib
from src.pilotlib.config import load_config, probe_imagenet
from src.pilotlib.embedders import QwenEmbedder
from src.embed_imagenet import _sample_paths

M=16
cfg=load_config("configs/pilot.yaml")
status=probe_imagenet(cfg.data.imagenet_root or None)
train=pathlib.Path(status.train_dir)
emb=QwenEmbedder(cfg.models.embedding, dtype=cfg.embedding.get("dtype","bfloat16"),
                 prompt=cfg.embedding.get("prompt",None))
seed=int(cfg.images.sampling_seed); bs=int(cfg.embedding.image_batch_size)
dirs=sorted(d for d in train.iterdir() if d.is_dir() and d.name.startswith("n"))
print(f"classes: {len(dirs)}", flush=True)
bank={}
for i,d in enumerate(dirs):
    paths=_sample_paths(d, M, seed)
    if not paths: continue
    try:
        v=emb.encode_image_paths(paths, batch_size=bs).mean(0)
    except Exception as e:
        print("skip", d.name, e); continue
    v=v/(np.linalg.norm(v)+1e-12)
    bank[d.name]=torch.from_numpy(v.astype(np.float32))
    if (i+1)%150==0: print(f"  {i+1}/{len(dirs)}", flush=True)
torch.save({"prototypes":bank,"dim":len(next(iter(bank.values()))),"M":M}, "cache/img_bank.pt")
print(f"wrote cache/img_bank.pt with {len(bank)} classes")
