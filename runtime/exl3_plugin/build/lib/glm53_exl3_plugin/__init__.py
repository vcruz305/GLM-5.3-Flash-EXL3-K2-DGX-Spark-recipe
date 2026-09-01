"""Register the routed-expert EXL3 implementation with local vLLM."""


def register() -> None:
    # Importing the module executes its register_quantization_config decorator.
    from . import exl3 as _exl3  # noqa: F401
