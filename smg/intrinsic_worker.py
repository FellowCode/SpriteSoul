"""Run in IntrinsicAnything's environment without importing the application."""

from pathlib import Path
import runpy
import sys


def inference_weights(state_dict):
    """Use the trained EMA UNet, preserving the other model weights and buffers."""
    import torch

    selected = {}
    used_ema = set()
    for name, tensor in state_dict.items():
        if name.startswith("model_ema."):
            continue
        ema_name = "model_ema." + name.removeprefix("model.").replace(".", "")
        if name.startswith("model.") and ema_name in state_dict:
            if ema_name in used_ema:
                raise RuntimeError(f"Ambiguous EMA parameter: {name}")
            tensor = state_dict[ema_name]
            used_ema.add(ema_name)
        # Own the small fp16 tensors so the training checkpoint can be unmapped.
        selected[name] = (tensor.to(dtype=torch.float16, copy=True)
                          if tensor.is_floating_point() else tensor.clone())
    expected_ema = {name for name in state_dict if name.startswith("model_ema.")}
    expected_ema -= {"model_ema.decay", "model_ema.num_updates"}
    if used_ema != expected_ema:
        raise RuntimeError("IntrinsicAnything: checkpoint contains unmatched EMA weights")
    return selected


def load_model(config, ckpt, device, vram_O=False, verbose=True):
    import gc
    import torch
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    print("IntrinsicAnything: loading inference weights…", flush=True)
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    weights = inference_weights(checkpoint["state_dict"])
    del checkpoint
    gc.collect()

    # Avoid allocating a second UNet for LitEma, and avoid fp32 model copies.
    inference_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    inference_config.model.params.use_ema = False
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        model = instantiate_from_config(inference_config.model)
    finally:
        torch.set_default_dtype(previous_dtype)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"IntrinsicAnything: missing model weights: {missing}")
    if unexpected and verbose:
        print(f"IntrinsicAnything: unused checkpoint keys: {unexpected}", flush=True)
    del weights
    if vram_O:
        del model.first_stage_model.decoder
    model.eval().half().to(device)
    for param in model.first_stage_model.parameters():
        param.requires_grad = True
    print("IntrinsicAnything: model ready", flush=True)
    return model


def cached_model_loader():
    """Retain only the inference weights across sequential worker requests."""
    model = None

    def load_once(*args, **kwargs):
        nonlocal model
        if model is None:
            model = load_model(*args, **kwargs)
        return model

    return load_once


def run(root: Path, arguments: list[str], loader=load_model) -> None:
    sys.path.insert(0, str(root))
    from models import matfusion

    matfusion.load_model_from_config = loader
    sys.argv = [str(root / "inference.py"), *arguments]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    if sys.argv[2:] == ["--serve"]:
        from model_session import serve

        loader = cached_model_loader()
        serve(lambda arguments: run(root, arguments, loader))
    else:
        run(root, sys.argv[2:])
