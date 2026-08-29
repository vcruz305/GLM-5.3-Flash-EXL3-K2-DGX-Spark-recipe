#!/usr/bin/env python3
"""Patch GLM-5.3 NoPE sparse MLA for the SM121 local vLLM branch.

The SM120 FlashInfer sparse-MLA kernel exposes the 576-wide GLM_NSA geometry,
while GLM-5.3-Flash stores a 512-wide NoPE latent. Appending an unused zero
64-wide RoPE lane preserves the dot product and satisfies the kernel ABI.
"""
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


def replace_one_of(path: Path, old_options: tuple[str, ...], new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if new in source:
        return
    matches = [(old, source.count(old)) for old in old_options]
    hits = [(old, count) for old, count in matches if count]
    if len(hits) != 1 or hits[0][1] != 1:
        raise RuntimeError(f"{path}: expected one migration target, found {matches}")
    path.write_text(source.replace(hits[0][0], new), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="vLLM source checkout")
    args = parser.parse_args()
    site = args.source / "vllm"

    sm120 = site / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
    replace_once(
        sm120,
        '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
        "        from vllm.config import get_current_vllm_config\n",
        '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
        "        self.rope_pad = 0\n"
        "        if self.qk_rope_head_dim == 0:\n"
        "            if self.kv_lora_rank != 512:\n"
        "                raise NotImplementedError(\n"
        '                    "NoPE MLA padding on SM120 requires "\n'
        '                    f"kv_lora_rank=512; got {self.kv_lora_rank}."\n'
        "                )\n"
        "            self.rope_pad = 64\n"
        "        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad\n"
        "        from vllm.config import get_current_vllm_config\n",
    )
    replace_once(
        sm120,
        "        if isinstance(q, tuple):\n"
        "            q = torch.cat(q, dim=-1)\n"
        "\n"
        "        num_actual_toks = q.shape[0]\n",
        "        if isinstance(q, tuple):\n"
        "            q = torch.cat(q, dim=-1)\n"
        "        if self.rope_pad:\n"
        "            q = torch.nn.functional.pad(q, (0, self.rope_pad))\n"
        "\n"
        "        num_actual_toks = q.shape[0]\n",
    )
    replace_once(
        sm120,
        "        topk_indices_physical = cast(\n"
        "            torch.Tensor,\n"
        "            triton_convert_req_index_to_global_index(\n"
        "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
        "                attn_metadata.block_table,\n"
        "                topk_indices,\n"
        "                BLOCK_SIZE=attn_metadata.block_size,\n"
        "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
        "            ),\n"
        "        )\n",
        "        topk_indices_physical, topk_lengths = cast(\n"
        "            tuple[torch.Tensor, torch.Tensor],\n"
        "            triton_convert_req_index_to_global_index(\n"
        "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
        "                attn_metadata.block_table,\n"
        "                topk_indices,\n"
        "                BLOCK_SIZE=attn_metadata.block_size,\n"
        "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
        "                return_valid_counts=True,\n"
        "            ),\n"
        "        )\n"
        "        sparse_topk_capacity = topk_indices_physical.shape[1]\n"
        "        empty_rows = topk_lengths == 0\n"
        "        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(\n"
        "            empty_rows, 0\n"
        "        )\n"
        "        topk_lengths = topk_lengths.clamp(min=1)\n",
    )
    replace_once(
        sm120,
        "            qk_rope_head_dim=self.qk_rope_head_dim,\n",
        "            qk_rope_head_dim=self.kernel_qk_rope_head_dim,\n",
    )
    replace_once(
        sm120,
        "            seq_lens=None,\n"
        "            max_seq_len=attn_metadata.topk_tokens,\n",
        "            seq_lens=topk_lengths,\n"
        "            max_seq_len=sparse_topk_capacity,\n",
    )
    replace_once(
        sm120,
        "            sparse_mla_top_k=attn_metadata.topk_tokens,\n",
        "            sparse_mla_top_k=sparse_topk_capacity,\n",
    )
    replace_once(
        sm120,
        "        return out.squeeze(1), None\n",
        "        out = out.squeeze(1)\n"
        "        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)\n"
        "        return out, None\n"
        "\n"
        "    def do_kv_cache_update(\n"
        "        self,\n"
        "        kv_c_normed: torch.Tensor,\n"
        "        k_pe: torch.Tensor,\n"
        "        kv_cache: torch.Tensor,\n"
        "        slot_mapping: torch.Tensor,\n"
        "        kv_cache_dtype: str,\n"
        "        k_scale: torch.Tensor,\n"
        "    ) -> None:\n"
        "        if self.rope_pad:\n"
        "            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))\n"
        "        super().do_kv_cache_update(\n"
        "            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n"
        "        )\n",
    )

    old_width = "buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)"
    for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
        replace_once(site / rel, old_width, "buffer_width = topk_tokens")

    sparse_backend = site / "v1/attention/backends/mla/flashinfer_mla_sparse.py"
    replace_once(
        sparse_backend,
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [64, 256]\n",
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [64]\n",
    )

    platform = site / "platforms/cuda.py"
    replace_once(
        platform,
        "        return index_kpool * min(PAGED_MQA_PAGE_SIZES)\n",
        "        page_sizes = PAGED_MQA_PAGE_SIZES\n"
        "        capability = cls.get_device_capability()\n"
        "        if capability is not None and capability.major == 12:\n"
        "            page_sizes = tuple(p for p in page_sizes if p == 64)\n"
        "        return index_kpool * min(page_sizes)\n",
    )
    replace_once(platform, "        return major >= 9\n", "        return major in (9, 10)\n")

    indexer = site / "model_executor/layers/sparse_attn_indexer_kpool.py"
    replace_once(
        indexer,
        "                    expanded = kpool_ops.expand_pools_and_append_tail(\n"
        "                        pool_ids, q_seq, index_kpool\n"
        "                    )\n",
        "                    expanded = kpool_ops.expand_pools_and_append_tail(\n"
        "                        pool_ids[:, : select_k - 1], q_seq, index_kpool\n"
        "                    )\n",
    )
    replace_once(
        indexer,
        "            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)\n",
        "            out = kpool_ops.expand_pools_and_append_tail(\n"
        "                pool_ids[:, : select_k - 1], dec_seq, index_kpool\n"
        "            )\n",
    )

    kv_cache_utils = site / "v1/core/kv_cache_utils.py"
    legacy_draft_groups = (
        "    elif glm5_groups := _get_kv_cache_groups_glm5_next(vllm_config, kv_cache_spec):\n"
        "        return glm5_groups\n"
        "\n"
        "    # A multi-module drafter adds regular attention and hidden-state cache\n"
        "    # specs to the target model's GLM Mamba/MLA/indexer specs. Preserve the\n"
        "    # specialized GLM grouping and group the drafter-owned caches on their\n"
        "    # own; the generic page unifier cannot pad sparse MLA index pages.\n"
        "    glm5_target_specs = {\n"
        "        name: spec\n"
        "        for name, spec in kv_cache_spec.items()\n"
        "        if isinstance(spec, (MambaSpec, KpoolTailSpec))\n"
        "        or type(spec) is MLAAttentionSpec\n"
        "    }\n"
        "    if glm5_target_specs and len(glm5_target_specs) != len(kv_cache_spec):\n"
        "        glm5_groups = _get_kv_cache_groups_glm5_next(\n"
        "            vllm_config, glm5_target_specs\n"
        "        )\n"
        "        if glm5_groups is not None:\n"
        "            draft_specs = {\n"
        "                name: spec\n"
        "                for name, spec in kv_cache_spec.items()\n"
        "                if name not in glm5_target_specs\n"
        "            }\n"
        "            return glm5_groups + get_kv_cache_groups(vllm_config, draft_specs)\n"
        "\n"
        "    # Pull HiddenStateCacheSpec layers out before the general multi-group\n"
    )
    upstream_draft_groups = (
        "    elif glm5_groups := _get_kv_cache_groups_glm5_next(vllm_config, kv_cache_spec):\n"
        "        return glm5_groups\n"
        "\n"
        "    # Pull HiddenStateCacheSpec layers out before the general multi-group\n"
    )
    aligned_draft_groups = (
        "    elif glm5_groups := _get_kv_cache_groups_glm5_next(vllm_config, kv_cache_spec):\n"
        "        return glm5_groups\n"
        "\n"
        "    # A multi-module drafter adds regular attention and hidden-state cache\n"
        "    # specs to GLM's specialized Mamba/MLA/indexer groups. Keep the target\n"
        "    # grouping, then pad only regular draft and hidden-state pages to the\n"
        "    # target page. This retains LBHNC without padding sparse MLA rows.\n"
        "    glm5_target_specs = {\n"
        "        name: spec\n"
        "        for name, spec in kv_cache_spec.items()\n"
        "        if isinstance(spec, (MambaSpec, KpoolTailSpec))\n"
        "        or type(spec) is MLAAttentionSpec\n"
        "    }\n"
        "    if glm5_target_specs and len(glm5_target_specs) != len(kv_cache_spec):\n"
        "        glm5_groups = _get_kv_cache_groups_glm5_next(\n"
        "            vllm_config, glm5_target_specs\n"
        "        )\n"
        "        if glm5_groups is not None:\n"
        "            target_mla_pages = {\n"
        "                spec.page_size_bytes\n"
        "                for spec in glm5_target_specs.values()\n"
        "                if type(spec) is MLAAttentionSpec\n"
        "                and spec.tokens_per_state == 1\n"
        "            }\n"
        "            if len(target_mla_pages) != 1:\n"
        "                raise ValueError(\n"
        "                    f\"expected one GLM target MLA page, got \"\n"
        "                    f\"{sorted(target_mla_pages)}\"\n"
        "                )\n"
        "            target_page = target_mla_pages.pop()\n"
        "            hidden_specs = {\n"
        "                name: spec\n"
        "                for name, spec in kv_cache_spec.items()\n"
        "                if isinstance(spec, HiddenStateCacheSpec)\n"
        "            }\n"
        "            draft_specs = {\n"
        "                name: spec\n"
        "                for name, spec in kv_cache_spec.items()\n"
        "                if name not in glm5_target_specs and name not in hidden_specs\n"
        "            }\n"
        "            padded_draft_specs: dict[str, KVCacheSpec] = {}\n"
        "            for name, spec in draft_specs.items():\n"
        "                if not isinstance(spec, AttentionSpec) or isinstance(\n"
        "                    spec, MLAAttentionSpec\n"
        "                ):\n"
        "                    raise ValueError(\n"
        "                        f\"unsupported GLM drafter cache spec {name}: {type(spec)}\"\n"
        "                    )\n"
        "                if spec.unpadded_page_size_bytes > target_page:\n"
        "                    raise ValueError(\n"
        "                        f\"draft cache page {name} ({spec.unpadded_page_size_bytes}) \"\n"
        "                        f\"does not fit GLM target page ({target_page})\"\n"
        "                    )\n"
        "                padded_draft_specs[name] = replace(\n"
        "                    spec, page_size_padded=target_page\n"
        "                )\n"
        "            draft_groups = _get_kv_cache_groups_uniform_page_size(\n"
        "                padded_draft_specs\n"
        "            )\n"
        "            group_block_size = math.gcd(\n"
        "                *(\n"
        "                    group.kv_cache_spec.block_size\n"
        "                    for group in glm5_groups + draft_groups\n"
        "                )\n"
        "            )\n"
        "            hidden_groups: list[KVCacheGroupSpec] = []\n"
        "            for name, spec in hidden_specs.items():\n"
        "                per_token = (\n"
        "                    spec.unpadded_page_size_bytes // spec.block_size\n"
        "                )\n"
        "                max_bs = max(target_page // per_token, 1)\n"
        "                hidden_bs = _largest_divisor_at_most(\n"
        "                    group_block_size, max_bs\n"
        "                )\n"
        "                hidden_spec = replace(\n"
        "                    spec, block_size=hidden_bs, page_size_padded=target_page\n"
        "                )\n"
        "                hidden_groups.append(KVCacheGroupSpec([name], hidden_spec))\n"
        "            return glm5_groups + draft_groups + hidden_groups\n"
        "\n"
        "    # Pull HiddenStateCacheSpec layers out before the general multi-group\n"
    )
    old_target_page = (
        "            target_page = get_uniform_page_size(\n"
        "                [group.kv_cache_spec for group in glm5_groups]\n"
        "            )\n"
    )
    old_hidden_per_token = (
        "                per_token = (\n"
        "                    spec.num_kv_heads\n"
        "                    * spec.head_size\n"
        "                    * get_dtype_size(spec.dtype)\n"
        "                )\n"
    )
    # A previous revision of this patch may already have installed the aligned
    # grouping with the wrong page-size anchor. Preserve that block so the
    # focused migration below can repair it in place.
    kv_source = kv_cache_utils.read_text(encoding="utf-8")
    if "    # A multi-module drafter adds regular attention" not in kv_source:
        replace_one_of(
            kv_cache_utils,
            (upstream_draft_groups, legacy_draft_groups),
            aligned_draft_groups,
        )
    if "draft_block_limit = int(" not in kv_cache_utils.read_text(encoding="utf-8"):
        replace_once(
            kv_cache_utils,
        old_target_page,
        "            target_mla_pages = {\n"
        "                spec.page_size_bytes\n"
        "                for spec in glm5_target_specs.values()\n"
        "                if type(spec) is MLAAttentionSpec\n"
        "                and spec.tokens_per_state == 1\n"
        "            }\n"
        "            if len(target_mla_pages) != 1:\n"
        "                raise ValueError(\n"
        "                    f\"expected one GLM target MLA page, got \"\n"
        "                    f\"{sorted(target_mla_pages)}\"\n"
        "                )\n"
        "            target_page = target_mla_pages.pop()\n",
    )
    replace_once(
        kv_cache_utils,
        old_hidden_per_token,
        "                per_token = (\n"
        "                    spec.unpadded_page_size_bytes // spec.block_size\n"
        "                )\n",
    )
    replace_one_of(
        kv_cache_utils,
        (
            "    # grouping, then pad only regular draft and hidden-state pages to the\n"
            "    # target page. This retains LBHNC without padding sparse MLA rows.\n",
            "    # grouping. Retile regular draft pages to the target manager block size;\n"
            "    # keeping a 16-token draft block beside a 9216-token target block would\n"
            "    # consume hundreds of enormous shared-pool block IDs per request.\n",
        ),
        "    # grouping. Retile regular draft pages to a configurable divisor of the\n"
        "    # target manager block; a 16-token draft block consumes hundreds of\n"
        "    # shared-pool IDs, while one full target-size draft page leaves too few.\n",
    )
    if "draft_block_limit = int(" not in kv_cache_utils.read_text(encoding="utf-8"):
        replace_once(
            kv_cache_utils,
        "            target_page = target_mla_pages.pop()\n"
        "            hidden_specs = {\n",
        "            target_page = target_mla_pages.pop()\n"
        "            target_block_sizes = {\n"
        "                group.kv_cache_spec.block_size\n"
        "                for group in glm5_groups\n"
        "                if isinstance(group.kv_cache_spec, MambaSpec)\n"
        "            }\n"
        "            if len(target_block_sizes) != 1:\n"
        "                raise ValueError(\n"
        "                    f\"expected one GLM manager block size, got \"\n"
        "                    f\"{sorted(target_block_sizes)}\"\n"
        "                )\n"
        "            target_block_size = target_block_sizes.pop()\n"
        "            draft_block_size = int(\n"
        "                os.environ.get(\n"
        "                    \"GLM_DFLASH_MANAGER_BLOCK_SIZE\", target_block_size\n"
        "                )\n"
        "            )\n"
        "            if (\n"
        "                draft_block_size < 128\n"
        "                or draft_block_size % 16\n"
        "                or target_block_size % draft_block_size\n"
        "            ):\n"
        "                raise ValueError(\n"
        "                    f\"GLM DFlash manager block {draft_block_size} must be \"\n"
        "                    f\">=128, divisible by 16, and divide target block \"\n"
        "                    f\"{target_block_size}\"\n"
        "                )\n"
            "            hidden_specs = {\n",
        )
    replace_one_of(
        kv_cache_utils,
        (
            "            draft_block_size = int(\n"
            "                os.environ.get(\n"
            "                    \"GLM_DFLASH_MANAGER_BLOCK_SIZE\", target_block_size\n"
            "                )\n"
            "            )\n"
            "            if (\n"
            "                draft_block_size < 128\n"
            "                or draft_block_size % 16\n"
            "                or target_block_size % draft_block_size\n"
            "            ):\n"
            "                raise ValueError(\n"
            "                    f\"GLM DFlash manager block {draft_block_size} must be \"\n"
            "                    f\">=128, divisible by 16, and divide target block \"\n"
            "                    f\"{target_block_size}\"\n"
            "                )\n",
            "            draft_block_limit = int(\n"
            "                os.environ.get(\n"
            "                    \"GLM_DFLASH_MANAGER_BLOCK_SIZE\", target_block_size\n"
            "                )\n"
            "            )\n"
            "            if draft_block_limit < 256 or draft_block_limit % 16:\n"
            "                raise ValueError(\n"
            "                    f\"GLM DFlash manager block limit {draft_block_limit} \"\n"
            "                    \"must be >=256 and divisible by 16\"\n"
            "                )\n"
            "            draft_block_size = _largest_divisor_at_most(\n"
            "                target_block_size, draft_block_limit\n"
            "            )\n"
            "            if draft_block_size < 256 or draft_block_size % 16:\n"
            "                raise ValueError(\n"
            "                    f\"no valid GLM DFlash manager block <= \"\n"
            "                    f\"{draft_block_limit} divides target block \"\n"
            "                    f\"{target_block_size}\"\n"
            "                )\n"
            "            logger.info(\n"
            "                \"GLM DFlash manager block: target=%d requested_max=%d \"\n"
            "                \"resolved=%d\",\n"
            "                target_block_size,\n"
            "                draft_block_limit,\n"
            "                draft_block_size,\n"
            "            )\n",
        ),
        "            draft_block_limit = int(\n"
        "                os.environ.get(\n"
        "                    \"GLM_DFLASH_MANAGER_BLOCK_SIZE\", target_block_size\n"
        "                )\n"
        "            )\n"
        "            if draft_block_limit < 128 or draft_block_limit % 16:\n"
        "                raise ValueError(\n"
        "                    f\"GLM DFlash manager block limit {draft_block_limit} \"\n"
        "                    \"must be >=128 and divisible by 16\"\n"
        "                )\n"
        "            draft_block_size = _largest_divisor_at_most(\n"
        "                target_block_size, draft_block_limit\n"
        "            )\n"
        "            if draft_block_size < 128 or draft_block_size % 16:\n"
        "                raise ValueError(\n"
        "                    f\"no valid GLM DFlash manager block <= \"\n"
        "                    f\"{draft_block_limit} divides target block \"\n"
        "                    f\"{target_block_size}\"\n"
        "                )\n"
        "            logger.info(\n"
        "                \"GLM DFlash manager block: target=%d requested_max=%d \"\n"
        "                \"resolved=%d\",\n"
        "                target_block_size,\n"
        "                draft_block_limit,\n"
        "                draft_block_size,\n"
        "            )\n",
    )
    replace_once(
        kv_cache_utils,
        "            padded_draft_specs: dict[str, KVCacheSpec] = {}\n"
        "            for name, spec in draft_specs.items():\n"
        "                if not isinstance(spec, AttentionSpec) or isinstance(\n"
        "                    spec, MLAAttentionSpec\n"
        "                ):\n"
        "                    raise ValueError(\n"
        "                        f\"unsupported GLM drafter cache spec {name}: {type(spec)}\"\n"
        "                    )\n"
        "                if spec.unpadded_page_size_bytes > target_page:\n"
        "                    raise ValueError(\n"
        "                        f\"draft cache page {name} ({spec.unpadded_page_size_bytes}) \"\n"
        "                        f\"does not fit GLM target page ({target_page})\"\n"
        "                    )\n"
        "                padded_draft_specs[name] = replace(\n"
        "                    spec, page_size_padded=target_page\n"
        "                )\n"
        "            draft_groups = _get_kv_cache_groups_uniform_page_size(\n"
        "                padded_draft_specs\n"
        "            )\n",
        "            retiled_draft_specs: dict[str, KVCacheSpec] = {}\n"
        "            for name, spec in draft_specs.items():\n"
        "                if not isinstance(spec, AttentionSpec) or isinstance(\n"
        "                    spec, MLAAttentionSpec\n"
        "                ):\n"
        "                    raise ValueError(\n"
        "                        f\"unsupported GLM drafter cache spec {name}: {type(spec)}\"\n"
        "                    )\n"
        "                if draft_block_size % spec.block_size:\n"
        "                    raise ValueError(\n"
        "                        f\"GLM draft manager block {draft_block_size} is not divisible \"\n"
        "                        f\"by draft block {spec.block_size} for {name}\"\n"
        "                    )\n"
        "                retiled_draft_specs[name] = replace(\n"
        "                    spec, block_size=draft_block_size, page_size_padded=None\n"
        "                )\n"
        "            draft_groups = _get_kv_cache_groups_uniform_page_size(\n"
        "                retiled_draft_specs\n"
        "            )\n",
    )
    replace_once(
        kv_cache_utils,
        "    if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):\n"
        "        return None\n"
        "\n",
        "    # Compatible auxiliary attention groups are validated below.\n"
        "\n",
    )
    replace_once(
        kv_cache_utils,
        "        if tail_page > idx_page:\n"
        "            return None\n"
        "\n"
        "    return (\n",
        "        if tail_page > idx_page:\n"
        "            return None\n"
        "\n"
        "    target_names = set(mla_names) | set(idx_names) | set(tail_names)\n"
        "    for group in mamba_groups:\n"
        "        target_names.update(group.layer_names)\n"
        "    for group in kv_cache_groups:\n"
        "        inside = [name in target_names for name in group.layer_names]\n"
        "        if any(inside) and not all(inside):\n"
        "            return None\n"
        "        if all(inside):\n"
        "            continue\n"
        "        group_spec = group.kv_cache_spec\n"
        "        extra_specs = (\n"
        "            group_spec.kv_cache_specs.values()\n"
        "            if isinstance(group_spec, UniformTypeKVCacheSpecs)\n"
        "            else (group_spec,)\n"
        "        )\n"
        "        if not all(\n"
        "            isinstance(spec, AttentionSpec)\n"
        "            and (\n"
        "                not isinstance(spec, MLAAttentionSpec)\n"
        "                or isinstance(spec, HiddenStateCacheSpec)\n"
        "            )\n"
        "            for spec in extra_specs\n"
        "        ):\n"
        "            return None\n"
        "\n"
        "    return (\n",
    )
    if "def _glm5_extra_groups(" not in kv_cache_utils.read_text(encoding="utf-8"):
        replace_once(
            kv_cache_utils,
            "def _get_kv_cache_bytes_per_block(\n",
            "def _glm5_extra_groups(\n"
            "    kv_cache_groups: list[KVCacheGroupSpec],\n"
            "    glm5_layout: tuple,\n"
            ") -> list[KVCacheGroupSpec]:\n"
            "    _, mamba_groups, mla_names, idx_names, _, _, tail_names, _ = glm5_layout\n"
            "    target_names = set(mla_names) | set(idx_names) | set(tail_names)\n"
            "    for group in mamba_groups:\n"
            "        target_names.update(group.layer_names)\n"
            "    return [\n"
            "        group\n"
            "        for group in kv_cache_groups\n"
            "        if group.layer_names\n"
            "        and not any(name in target_names for name in group.layer_names)\n"
            "    ]\n"
            "\n"
            "\n"
            "def _get_kv_cache_bytes_per_block(\n",
        )
    replace_once(
        kv_cache_utils,
        "        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout\n"
        "        return len(mla_names) * mla_page + len(idx_names) * idx_page\n",
        "        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout\n"
        "        target_bytes = len(mla_names) * mla_page + len(idx_names) * idx_page\n"
        "        extra_bytes = sum(\n"
        "            sum(\n"
        "                _get_per_layer_spec(group, name).page_size_bytes\n"
        "                for name in group.layer_names\n"
        "            )\n"
        "            for group in _glm5_extra_groups(kv_cache_groups, glm5_layout)\n"
        "        )\n"
        "        return target_bytes + extra_bytes\n",
    )
    replace_once(
        kv_cache_utils,
        "        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page\n"
        "        num_blocks = may_override_num_blocks(\n",
        "        bytes_per_block = _get_kv_cache_bytes_per_block(kv_cache_groups)\n"
        "        num_blocks = may_override_num_blocks(\n",
    )
    replace_once(
        kv_cache_utils,
        "        size = bytes_per_block * num_blocks\n"
        "        attn_specs = cast(\n",
        "        size = bytes_per_block * num_blocks\n"
        "        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()\n"
        "        if layout is not KVCacheLayout.LBHNC:\n"
        "            raise ValueError(\n"
        "                f\"GLM-5.3 sparse MLA packed allocation requires LBHNC; \"\n"
        "                f\"resolved {layout.name}\"\n"
        "            )\n"
        "        attn_specs = cast(\n",
    )
    if "GLM5 KV group %d" not in kv_cache_utils.read_text(encoding="utf-8"):
        replace_once(
            kv_cache_utils,
            "                ).kv_cache_specs\n"
        "                add_tensor(tail_name, tail_specs[tail_name], offset)\n"
        "\n"
        "        return KVCacheConfig(\n",
        "                ).kv_cache_specs\n"
        "                add_tensor(tail_name, tail_specs[tail_name], offset)\n"
        "\n"
        "        target_bytes_per_block = (\n"
        "            len(mla_names) * mla_page + len(idx_names) * idx_page\n"
        "        )\n"
        "        extra_base_per_block = target_bytes_per_block\n"
        "        for group in _glm5_extra_groups(kv_cache_groups, glm5_layout):\n"
        "            group_spec = group.kv_cache_spec\n"
        "            layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)\n"
        "            if isinstance(group_spec, UniformTypeKVCacheSpecs):\n"
        "                for layer_name, spec in group_spec.kv_cache_specs.items():\n"
        "                    layers_by_spec[spec].append(layer_name)\n"
        "            else:\n"
        "                layers_by_spec[group_spec].extend(group.layer_names)\n"
        "            byte_offset = 0\n"
        "            group_bytes_per_block = sum(\n"
        "                _get_per_layer_spec(group, name).page_size_bytes\n"
        "                for name in group.layer_names\n"
        "            )\n"
        "            group_base = (\n"
        "                extra_base_per_block * num_blocks\n"
        "                if layout.is_layer_compact\n"
        "                else extra_base_per_block\n"
        "            )\n"
        "            for spec, layer_names in layers_by_spec.items():\n"
        "                layer_stride, block_stride, _, _, _ = compute_layout_strides(\n"
        "                    spec, num_blocks, len(layer_names), layout\n"
        "                )\n"
        "                kv_cache_tensors.append(\n"
        "                    KVCacheTensor(\n"
        "                        size=size,\n"
        "                        layers=layer_names,\n"
        "                        layer_stride=layer_stride,\n"
        "                        block_stride=block_stride,\n"
        "                        offset=group_base\n"
        "                        + byte_offset\n"
        "                        * max(layer_stride, spec.page_size_bytes)\n"
        "                        // spec.page_size_bytes,\n"
        "                    )\n"
        "                )\n"
        "                byte_offset += spec.page_size_bytes * len(layer_names)\n"
        "            assert byte_offset == group_bytes_per_block\n"
        "            extra_base_per_block += group_bytes_per_block\n"
        "        assert extra_base_per_block == bytes_per_block\n"
        "\n"
            "        return KVCacheConfig(\n",
        )
    replace_once(
        kv_cache_utils,
        "        assert extra_base_per_block == bytes_per_block\n"
        "\n"
        "        return KVCacheConfig(\n",
        "        assert extra_base_per_block == bytes_per_block\n"
        "        for group_index, group in enumerate(kv_cache_groups):\n"
        "            group_spec = group.kv_cache_spec\n"
        "            if isinstance(group_spec, UniformTypeKVCacheSpecs):\n"
        "                group_specs = tuple(group_spec.kv_cache_specs.values())\n"
        "            else:\n"
        "                group_specs = (group_spec,)\n"
        "            logger.info(\n"
        "                \"GLM5 KV group %d: layers=%d types=%s manager_blocks=%s \"\n"
        "                \"pages=%s shared_pool_blocks=%d\",\n"
        "                group_index,\n"
        "                len(group.layer_names),\n"
        "                sorted({type(spec).__name__ for spec in group_specs}),\n"
        "                sorted({spec.block_size for spec in group_specs}),\n"
        "                sorted({spec.page_size_bytes for spec in group_specs}),\n"
        "                num_blocks,\n"
        "            )\n"
        "\n"
        "        return KVCacheConfig(\n",
    )
    replace_once(
        kv_cache_utils,
        "        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)\n",
        "        total_blocks += sum(\n"
        "            cdiv(\n"
        "                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),\n"
        "                group.kv_cache_spec.page_size_bytes,\n"
        "            )\n"
        "            for group in _glm5_extra_groups(kv_cache_groups, glm5_layout)\n"
        "        )\n"
        "        return total_blocks * _get_kv_cache_bytes_per_block(kv_cache_groups)\n",
    )

    kv_source = kv_cache_utils.read_text(encoding="utf-8")
    required_kv_markers = (
        "GLM_DFLASH_MANAGER_BLOCK_SIZE",
        "draft_block_limit = int(",
        "draft_block_size = _largest_divisor_at_most(",
        "GLM5 KV group %d:",
        "extra_base_per_block += group_bytes_per_block",
    )
    missing_markers = [marker for marker in required_kv_markers if marker not in kv_source]
    if missing_markers:
        raise RuntimeError(
            f"{kv_cache_utils}: missing GLM/DFlash postconditions: {missing_markers}"
        )

    compile(sm120.read_text(encoding="utf-8"), str(sm120), "exec")
    for rel in (
        "models/glm5next/nvidia/model.py",
        "models/glm5next/nvidia/mtp.py",
        "model_executor/layers/sparse_attn_indexer_kpool.py",
        "v1/attention/backends/mla/flashinfer_mla_sparse.py",
        "platforms/cuda.py",
        "v1/core/kv_cache_utils.py",
    ):
        path = site / rel
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("GLM-5.3 NoPE sparse-MLA SM121 source patch verified")


if __name__ == "__main__":
    main()
