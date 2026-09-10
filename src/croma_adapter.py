from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch


class CROMAAdapter:
    def __init__(self, source: Path, checkpoint: Path, device: str | None = None):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Official CROMA checkpoint not found: {checkpoint}")
        use_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(use_device)
        module_path = source / "use_croma.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Official CROMA implementation not found: {module_path}")
        spec = importlib.util.spec_from_file_location("official_croma", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load official CROMA module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["official_croma"] = module
        spec.loader.exec_module(module)
        self.model = module.PretrainedCROMA(
            pretrained_path=str(checkpoint), size="base", modality="both", image_resolution=120
        ).to(self.device).eval()

    @staticmethod
    def _croma_normalize(array: np.ndarray) -> torch.Tensor:
        """Match the official README: raw channels -> mean +/- 2 std -> [0, 1]."""
        tensor = torch.from_numpy(array).float()
        normalized = []
        for channel in tensor:
            mean = channel.mean()
            std = channel.std()
            lower, upper = mean - 2 * std, mean + 2 * std
            if float(upper - lower) == 0:
                normalized.append(torch.zeros_like(channel))
            else:
                normalized.append(torch.clamp((channel - lower) / (upper - lower), 0, 1))
        return torch.stack(normalized).unsqueeze(0)

    @torch.no_grad()
    def infer(self, optical: np.ndarray, sar: np.ndarray) -> dict[str, torch.Tensor]:
        if optical.shape != (12, 120, 120) or sar.shape != (2, 120, 120):
            raise ValueError(f"CROMA requires [12,120,120] and [2,120,120], got {optical.shape} and {sar.shape}")
        outputs = self.model(
            SAR_images=self._croma_normalize(sar).to(self.device),
            optical_images=self._croma_normalize(optical).to(self.device),
        )
        return outputs

    @torch.no_grad()
    def infer_modality(self, optical: np.ndarray | None = None, sar: np.ndarray | None = None) -> dict[str, torch.Tensor]:
        """Use the official encoders independently, without manufacturing a missing input."""
        if optical is not None and sar is not None:
            return self.infer(optical, sar)
        array, name, bands = (optical, "optical", 12) if optical is not None else (sar, "SAR", 2)
        if array is None or array.shape != (bands, 120, 120) or not np.isfinite(array).all():
            raise ValueError("CROMA requires finite canonical 120x120 modality tensors")
        images = self._croma_normalize(array).to(self.device)
        encoder = self.model.s2_encoder if name == "optical" else self.model.s1_encoder
        pool = self.model.GAP_FFN_s2 if name == "optical" else self.model.GAP_FFN_s1
        encodings = encoder(imgs=images, attn_bias=self.model.attn_bias.to(self.device))
        return {f"{name}_encodings": encodings, f"{name}_GAP": pool(encodings.mean(dim=1))}
