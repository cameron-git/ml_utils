import torch

__all__ = ["Cache", "DynamicCache", "StaticCache"]


class Cache:
    def __getitem__(self, key) -> torch.Tensor:
        raise NotImplementedError

    def update(self, key: str, value: torch.Tensor, dim=-2) -> None:
        raise NotImplementedError


class DynamicCache(Cache):
    def __init__(self):
        super().__init__()
        self.cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self.cache:
            raise KeyError(f"Key {key} not found in cache")
        return self.cache[key]

    def update(self, key: str, value: torch.Tensor, dim=-2) -> None:
        if key in self.cache:
            self.cache[key] = torch.cat([self.cache[key], value], dim=dim)
        else:
            self.cache[key] = value

        return self.cache[key]

    def __len__(self) -> int:
        return self.cache[list(self.cache.keys())[0]].shape[-2] if self.cache else 0


# TODO: keep track of current position in cache
# TODO: implement cache update
class StaticCache(Cache):
    def __init__(self, cache: dict[str, torch.Tensor]):
        super().__init__()
        self.cache = cache

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self.cache:
            raise KeyError(f"Key {key} not found in cache")
        return self.cache[key]

    def update(self) -> None:
        pass
