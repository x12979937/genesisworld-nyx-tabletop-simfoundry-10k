#!/usr/bin/env bash
set -euo pipefail

ROOT="/autodl-fs/data/mingyu/video2sim_roboticArm/datasets/genesisworld-nyx-tabletop-simfoundry-10k"
STATUS="$ROOT/status.json"
UP_STATUS="$ROOT/github/uploader_status.json"
LOGDIR="$ROOT/logs"
INTERVAL="${1:-30}"
PY="/root/autodl-tmp/mingyu/venvs/genesisworld_py310/bin/python"
[[ -x "$PY" ]] || PY="python3"

while true; do
  if [[ -n "${TERM:-}" ]]; then
    clear || true
  fi
  date '+%F %T %Z'
  echo
  echo "== processes =="
  ps -eo pid,ppid,stat,etimes,cmd | grep -E 'run_geniesim_10k_dataset|geniesim_tabletop_state_dataset_demo|upload_geniesim_dataset_shards' | grep -v grep || true
  echo
  echo "== disk =="
  df -h /root/autodl-tmp /autodl-fs/data || true
  echo
  df -ih /root/autodl-tmp /autodl-fs/data || true
  echo
  echo "== generation status =="
  if [[ -f "$STATUS" ]]; then
    "$PY" - <<'PY' "$STATUS"
import json, sys
p=sys.argv[1]
d=json.load(open(p, encoding='utf-8'))
for k in ['state','accepted_count','target_total','completed_batches','failed_batches','next_group_id','next_batch_seq','github_upload_enabled','updated_at_utc']:
    if k in d:
        print(f'{k}: {d[k]}')
if d.get('in_flight'):
    print('in_flight:', d['in_flight'])
if 'last_completed_batch' in d:
    print('last_completed_batch:', d['last_completed_batch'])
if 'last_error' in d:
    print('last_error:', d['last_error'])
PY
  else
    echo "status not created yet: $STATUS"
  fi
  echo
  echo "== uploader status =="
  if [[ -f "$UP_STATUS" ]]; then
    "$PY" - <<'PY' "$UP_STATUS"
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
for k in ['state','repo_url','release_tag','archives_total','uploaded_recorded','pending_count','uploaded_this_pass','current_archive','current_size_bytes','updated_at_utc']:
    if k in d:
        print(f'{k}: {d[k]}')
if d.get('pending'):
    print('pending_head:', d['pending'])
if d.get('last_error'):
    print('last_error:', d['last_error'])
PY
  else
    echo "uploader status not created yet: $UP_STATUS"
  fi
  echo
  echo "== latest archives =="
  ls -lh "$ROOT/archives" 2>/dev/null | tail -10 || true
  echo
  echo "== recent errors =="
  tail -5 "$LOGDIR/batch_errors.jsonl" 2>/dev/null || true
  tail -5 "$ROOT/github/upload_errors.jsonl" 2>/dev/null || true
  echo
  echo "== latest batch log tail =="
  latest="$(ls -t "$LOGDIR"/batch_*.log 2>/dev/null | head -1 || true)"
  if [[ -n "$latest" ]]; then
    echo "$latest"
    tail -40 "$latest" || true
  else
    echo "no batch logs yet"
  fi
  echo
  echo "== runner/uploader nohup tail =="
  tail -10 "$LOGDIR/runner_10k.nohup.log" 2>/dev/null || true
  tail -10 "$LOGDIR/uploader.nohup.log" 2>/dev/null || true
  sleep "$INTERVAL"
done
