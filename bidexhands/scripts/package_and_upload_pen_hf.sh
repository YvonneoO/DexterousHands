#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/lp-dev/qianqian/DexterousHands/runs/tactile_dataset/shadow_hand_pen}"
STAGING_ROOT="${STAGING_ROOT:-/lp-dev/qianqian/DexterousHands/runs/hf_upload/sim-pen}"
HF_BIN="${HF_BIN:-/lp-dev/qianqian/envs/hf_upload/bin/hf}"
REPO_ID="${REPO_ID:-qqyang/sim-pen}"
MAX_BYTES=$((45 * 1024 * 1024 * 1024))

mkdir -p "$STAGING_ROOT"
rm -f "$STAGING_ROOT/UPLOAD_COMPLETE_UTC.txt"

cat > "$STAGING_ROOT/README.md" <<'EOF'
---
license: other
task_categories:
- reinforcement-learning
tags:
- dexterous-manipulation
- tactile
- rgb
- isaac-gym
---

# ShadowHandPen RGB + raw rigid-contact tactile trajectories

This dataset contains 5,000 successful `ShadowHandPen` PPO rollouts. Each
episode stores synchronized RGB frames, robot hand joint/object state in
`trajectory_env0.npz`, and dense EgoTouch-layout pressure in
`pressure_grids.npz`. The pressure grids are computed from pairwise rigid
contacts and Gaussian projection onto 217 valid taxels per hand.

The tar files preserve the original collection shards. Shard-level duplicate
RGB frame folders and QA-only videos are intentionally omitted.
EOF

printf 'archive\tbytes\tsha256\n' > "$STAGING_ROOT/SHA256SUMS.tsv"

for shard in "$DATA_ROOT"/*; do
  [[ -d "$shard/successful_episodes" ]] || continue
  name="$(basename "$shard")"
  archive="$STAGING_ROOT/${name}.tar"
  if [[ ! -s "$archive" ]]; then
    members=(successful_episodes summary.json)
    [[ -f "$shard/coverage_qa_report.json" ]] && members+=(coverage_qa_report.json)
    [[ -f "$shard/collector.log" ]] && members+=(collector.log)
    tar -cf "$archive" -C "$shard" "${members[@]}"
  fi
  bytes="$(stat -c %s "$archive")"
  if (( bytes > MAX_BYTES )); then
    echo "ERROR: $archive exceeds 45 GiB ($bytes bytes)" >&2
    exit 2
  fi
  digest="$(sha256sum "$archive" | awk '{print $1}')"
  printf '%s\t%s\t%s\n' "$(basename "$archive")" "$bytes" "$digest" >> "$STAGING_ROOT/SHA256SUMS.tsv"
done

"$HF_BIN" upload-large-folder "$REPO_ID" "$STAGING_ROOT" --repo-type dataset

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$STAGING_ROOT/UPLOAD_COMPLETE_UTC.txt"
"$HF_BIN" upload "$REPO_ID" "$STAGING_ROOT/UPLOAD_COMPLETE_UTC.txt" UPLOAD_COMPLETE_UTC.txt --repo-type dataset
