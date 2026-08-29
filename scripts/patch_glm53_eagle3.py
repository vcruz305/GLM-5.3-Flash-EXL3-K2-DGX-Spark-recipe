#!/usr/bin/env python3
"""Add GLM-5.3 auxiliary-hidden-state taps required by DFlash2."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count == 1:
        path.write_text(source.replace(old, new), encoding="utf-8")
        return
    if count == 0 and new in source:
        return
    raise RuntimeError(f"{path}: expected one patch target, found {count}: {old!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="vLLM source checkout")
    args = parser.parse_args()
    target = args.source / "vllm/models/glm5next/nvidia/model.py"

    replace_once(
        target,
        "from vllm.model_executor.models.interfaces import (\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    MixtureOfExperts,\n"
        "    SupportsPP,\n"
        ")\n",
        "from vllm.model_executor.models.interfaces import (\n"
        "    EagleModelMixin,\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    MixtureOfExperts,\n"
        "    SupportsEagle3,\n"
        "    SupportsPP,\n"
        ")\n",
    )
    replace_once(
        target,
        "class Glm5NextModel(nn.Module):\n",
        "class Glm5NextModel(nn.Module, EagleModelMixin):\n",
    )
    replace_once(
        target,
        "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n",
        "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n"
        "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n",
    )
    replace_once(
        target,
        "        full_num_tokens = positions.shape[0]\n"
        "        if self.is_sequence_parallel:\n"
        "            hidden_states = sp_shard(hidden_states)\n"
        "\n"
        "        for layer in self._active_layers:\n"
        "            hidden_states, residual, post, comb = layer(\n"
        "                positions, hidden_states, residual, post, comb\n"
        "            )\n",
        "        full_num_tokens = positions.shape[0]\n"
        "        if self.is_sequence_parallel:\n"
        "            hidden_states = sp_shard(hidden_states)\n"
        "\n"
        "        aux_hidden_states: list[torch.Tensor] = []\n"
        "        for idx, layer in enumerate(\n"
        "            self._active_layers, start=self.start_layer\n"
        "        ):\n"
        "            hidden_states, residual, post, comb = layer(\n"
        "                positions, hidden_states, residual, post, comb\n"
        "            )\n"
        "            if idx + 1 not in self.aux_hidden_state_layers:\n"
        "                continue\n"
        "            # Mid-stack mHC defers hc_post. Materialize and contract\n"
        "            # its four streams to the 4096-wide state used to train DFlash2.\n"
        "            if post is not None and hasattr(layer, \"hc_post\"):\n"
        "                value = hc_contract(\n"
        "                    layer.hc_post(hidden_states, residual, post, comb),\n"
        "                    layer.n,\n"
        "                )\n"
        "            else:\n"
        "                value = hidden_states\n"
        "                if value.ndim == 3:\n"
        "                    value = value.mean(dim=1)\n"
        "            if self.is_sequence_parallel:\n"
        "                value = sp_all_gather(value)[:full_num_tokens]\n"
        "            aux_hidden_states.append(value)\n",
    )
    replace_once(
        target,
        "        hidden_states = self.norm(hidden_states)\n"
        "        return hidden_states\n",
        "        hidden_states = self.norm(hidden_states)\n"
        "        if aux_hidden_states:\n"
        "            return hidden_states, aux_hidden_states\n"
        "        return hidden_states\n",
    )
    replace_once(
        target,
        "class Glm5NextForCausalLM(\n"
        "    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid\n"
        "):\n",
        "class Glm5NextForCausalLM(\n"
        "    nn.Module,\n"
        "    HasInnerState,\n"
        "    SupportsPP,\n"
        "    MixtureOfExperts,\n"
        "    IsHybrid,\n"
        "    SupportsEagle3,\n"
        "):\n",
    )
    replace_once(
        target,
        "class Glm5NextForConditionalGeneration(\n"
        "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid\n"
        "):\n",
        "class Glm5NextForConditionalGeneration(\n"
        "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3\n"
        "):\n",
    )

    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    print("GLM-5.3 EAGLE3 auxiliary-state source patch verified")


if __name__ == "__main__":
    main()
