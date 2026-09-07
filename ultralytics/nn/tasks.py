import contextlib
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            raise TypeError("Dictionary input is not supported in the minimal inference package.")
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            raise NotImplementedError("augment=True is not supported in the minimal inference package.")
        return self._predict_once(x, profile=profile, visualize=visualize, embed=embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, embeddings = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if embed and m.i in embed:
                pooled = nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                embeddings.append(pooled)
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x


class DetectionModel(BaseModel):
    pass


@contextlib.contextmanager
def temporary_modules(modules=None):
    if modules is None:
        modules = {}
    import sys
    from importlib import import_module

    try:
        for old, new in modules.items():
            sys.modules[old] = import_module(new)
        yield
    finally:
        for old in modules:
            sys.modules.pop(old, None)


def torch_safe_load(weight):
    weight = str(weight)
    if not Path(weight).exists():
        raise FileNotFoundError(f"Model not found: {weight}")

    with temporary_modules(
        modules={
            "ultralytics.yolo.utils": "ultralytics.utils",
            "ultralytics.yolo.v8": "ultralytics.nn",
            "ultralytics.yolo.data": "ultralytics.nn",
        }
    ):
        try:
            ckpt = torch.load(weight, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(weight, map_location="cpu")

    if not isinstance(ckpt, dict):
        ckpt = {"model": ckpt}
    return ckpt, weight


def guess_model_task(model):
    return "detect"


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    ckpt, weight = torch_safe_load(weight)
    model = (ckpt.get("ema") or ckpt["model"]).to(device or torch.device("cpu")).float()
    model.args = getattr(model, "args", {})
    model.pt_path = weight
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None

    return model.eval(), ckpt

