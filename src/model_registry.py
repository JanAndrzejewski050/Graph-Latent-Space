from src.config import TrainConfig


_REGISTRY: dict[str, type] = {}


def register(name: str):
    def _decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return _decorator


def create_model(name: str, config: TrainConfig):
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown architecture '{name}'. Available: {available}"
        )
    return _REGISTRY[name](config)


def list_architectures() -> list[str]:
    return sorted(_REGISTRY.keys())
