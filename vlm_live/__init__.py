"""Support modules for the Panda VLM live verifier."""

__all__ = ["main", "VlmNode"]


def __getattr__(name):
    if name == "main":
        from .cli import main

        return main
    if name == "VlmNode":
        from .node import VlmNode

        return VlmNode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
