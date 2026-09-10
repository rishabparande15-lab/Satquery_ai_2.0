from src.croma_adapter import CROMAAdapter


class ExperimentalCROMAAdapter:
    """Safe boundary around the verified project adapter; it does not copy weights."""

    def __init__(self, source, checkpoint, device=None):
        self._adapter = CROMAAdapter(source, checkpoint, device=device)

    @property
    def device(self):
        return self._adapter.device

    def infer(self, optical, sar):
        return self._adapter.infer(optical, sar)
