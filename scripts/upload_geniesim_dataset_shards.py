#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path('/autodl-fs/data/mingyu/video2sim_roboticArm/GenesisWorld')
if str(PROJECT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from run_geniesim_10k_dataset import (  # noqa: E402
    DATASET_NAME,
    FS_ROOT,
    GithubClient,
    build_readme,
    utc_now,
    write_json_atomic,
)


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')


def read_uploaded_names(path):
    names = set()
    if not path.is_file():
        return names
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            archive = rec.get('archive')
            action = rec.get('action')
            if archive and action in {'uploaded', 'already_exists'}:
                names.add(archive)
    return names


def write_status(path, payload):
    payload = {**payload, 'updated_at_utc': utc_now()}
    write_json_atomic(path, payload)


def tail_text(path, max_lines=200):
    if not path.is_file():
        return None
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines(True)
    return ''.join(lines[-max_lines:])


def sync_repo_manifests(gh, args):
    script_dir = PROJECT_ROOT / 'scripts'
    files = [
        ('README.md', build_readme(args, getattr(args, 'repo_url', None)), 'Update dataset README'),
        ('scripts/run_geniesim_10k_dataset.py', (script_dir / 'run_geniesim_10k_dataset.py').read_text(encoding='utf-8'), 'Update dataset runner'),
        ('scripts/upload_geniesim_dataset_shards.py', Path(__file__).read_text(encoding='utf-8'), 'Update dataset uploader'),
    ]
    monitor = script_dir / 'monitor_geniesim_10k_dataset.sh'
    if monitor.is_file():
        files.append(('scripts/monitor_geniesim_10k_dataset.sh', monitor.read_text(encoding='utf-8'), 'Update dataset monitor'))
    status = FS_ROOT / 'status.json'
    if status.is_file():
        files.append(('manifests/status.json', status.read_text(encoding='utf-8'), 'Update dataset status'))
    uploads = FS_ROOT / 'github' / 'uploads.jsonl'
    if uploads.is_file():
        files.append(('manifests/uploads.jsonl', uploads.read_text(encoding='utf-8'), 'Update upload manifest'))
    upload_errors = FS_ROOT / 'github' / 'upload_errors.jsonl'
    if upload_errors.is_file():
        files.append(('manifests/upload_errors.jsonl', upload_errors.read_text(encoding='utf-8'), 'Update upload error manifest'))
    uploader_status = FS_ROOT / 'github' / 'uploader_status.json'
    if uploader_status.is_file():
        files.append(('manifests/uploader_status.json', uploader_status.read_text(encoding='utf-8'), 'Update uploader status'))
    global_head = tail_text(FS_ROOT / 'manifests' / 'global_manifest.jsonl')
    if global_head is not None:
        files.append(('manifests/global_manifest.head.jsonl', global_head, 'Update manifest tail'))
    batch_head = tail_text(FS_ROOT / 'manifests' / 'batch_results.jsonl')
    if batch_head is not None:
        files.append(('manifests/batch_results.head.jsonl', batch_head, 'Update batch result tail'))
    for path, content, message in files:
        gh.put_file(path, content, message)


def upload_once(args, gh):
    archives_dir = FS_ROOT / 'archives'
    uploads_path = FS_ROOT / 'github' / 'uploads.jsonl'
    errors_path = FS_ROOT / 'github' / 'upload_errors.jsonl'
    status_path = FS_ROOT / 'github' / 'uploader_status.json'
    uploaded_names = read_uploaded_names(uploads_path)
    archives = sorted(archives_dir.glob('*.tar.gz'))
    pending = [p for p in archives if p.name not in uploaded_names]
    write_status(status_path, {
        'state': 'idle' if not pending else 'uploading',
        'repo_url': getattr(args, 'repo_url', None),
        'release_tag': args.release_tag,
        'archives_total': len(archives),
        'uploaded_recorded': len(uploaded_names),
        'pending': [p.name for p in pending[:20]],
        'pending_count': len(pending),
    })
    uploaded_this_pass = 0
    for archive in pending:
        try:
            write_status(status_path, {
                'state': 'uploading',
                'repo_url': getattr(args, 'repo_url', None),
                'release_tag': args.release_tag,
                'current_archive': archive.name,
                'current_size_bytes': archive.stat().st_size,
                'pending_count': len(pending),
            })
            url, action = gh.upload_asset(archive, tag=args.release_tag)
            rec = {
                'archive': archive.name,
                'url': url,
                'action': action,
                'uploaded_at_utc': utc_now(),
                'size_bytes': archive.stat().st_size,
            }
            append_jsonl(uploads_path, rec)
            uploaded_names.add(archive.name)
            uploaded_this_pass += 1
            if not args.no_manifest_push:
                sync_repo_manifests(gh, args)
        except Exception as exc:
            err = {
                'archive': archive.name,
                'action': 'upload_failed',
                'error': repr(exc),
                'uploaded_at_utc': utc_now(),
            }
            append_jsonl(errors_path, err)
            write_status(status_path, {
                'state': 'error',
                'repo_url': getattr(args, 'repo_url', None),
                'release_tag': args.release_tag,
                'last_error': err,
                'pending_count': len(pending),
            })
            if not args.continue_on_error:
                raise
            time.sleep(args.error_sleep_seconds)
    write_status(status_path, {
        'state': 'idle',
        'repo_url': getattr(args, 'repo_url', None),
        'release_tag': args.release_tag,
        'archives_total': len(archives),
        'uploaded_recorded': len(uploaded_names),
        'uploaded_this_pass': uploaded_this_pass,
        'pending_count': max(0, len(archives) - len(uploaded_names)),
    })
    return uploaded_this_pass, len(pending)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--github-owner', default='x12979937')
    ap.add_argument('--github-repo', default=DATASET_NAME)
    ap.add_argument('--github-token-env', default='GITHUB_TOKEN')
    ap.add_argument('--github-private', action='store_true')
    ap.add_argument('--release-tag', default='dataset-shards-v1')
    ap.add_argument('--total', type=int, default=10000)
    ap.add_argument('--poll-seconds', type=int, default=120)
    ap.add_argument('--continue-on-error', action='store_true')
    ap.add_argument('--error-sleep-seconds', type=int, default=180)
    ap.add_argument('--no-manifest-push', action='store_true')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()

    token = os.environ.get(args.github_token_env)
    if not token:
        raise RuntimeError(f'{args.github_token_env} is not set')
    gh = GithubClient(args.github_owner, args.github_repo, token, private=args.github_private)
    repo_url = gh.ensure_repo()
    args.repo_url = repo_url
    sync_repo_manifests(gh, args)
    gh.ensure_release(args.release_tag)
    while True:
        upload_once(args, gh)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
