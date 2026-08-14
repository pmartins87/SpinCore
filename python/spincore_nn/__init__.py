from __future__ import annotations

from importlib import import_module

__all__ = [
    "NetworkConfig",
    "AdvantageNet",
    "AveragePolicyNet",
    "UniformReservoir",
    "AdvantageSample",
    "StrategySample",
    "DomainBundle",
]

_EXPORTS = {
    "NetworkConfig": (".models", "NetworkConfig"),
    "AdvantageNet": (".models", "AdvantageNet"),
    "AveragePolicyNet": (".models", "AveragePolicyNet"),
    "UniformReservoir": (".reservoir", "UniformReservoir"),
    "AdvantageSample": (".reservoir", "AdvantageSample"),
    "StrategySample": (".reservoir", "StrategySample"),
    "DomainBundle": (".bundle", "DomainBundle"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
