from src.croma_adapter import CROMAAdapter


def load_adapter(source, checkpoint):
    return CROMAAdapter(source, checkpoint)