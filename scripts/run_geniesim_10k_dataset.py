#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path('/autodl-fs/data/mingyu/video2sim_roboticArm/GenesisWorld')
GEN_SCRIPT = PROJECT_ROOT / 'scripts' / 'geniesim_tabletop_state_dataset_demo.py'
ENV_RUN = PROJECT_ROOT / 'scripts' / 'run_with_genesisworld_env.sh'

DATASET_NAME = 'genesisworld-nyx-tabletop-simfoundry-10k'
TMP_ROOT = Path('/root/autodl-tmp/mingyu/genesisworld/datasets') / DATASET_NAME
FS_ROOT = Path('/autodl-fs/data/mingyu/video2sim_roboticArm/datasets') / DATASET_NAME

GITHUB_API = 'https://api.github.com'
GITHUB_UPLOADS = 'https://uploads.github.com'
MAX_RELEASE_ASSET_BYTES = 2 * 1024 * 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def ensure_dirs(args):
    for p in [
        TMP_ROOT / 'raw_batches',
        TMP_ROOT / 'staging',
        TMP_ROOT / 'tmp',
        FS_ROOT / 'archives',
        FS_ROOT / 'logs',
        FS_ROOT / 'manifests',
        FS_ROOT / 'github',
    ]:
        p.mkdir(parents=True, exist_ok=True)


def append_jsonl(path, record, lock=None):
    if lock:
        lock.acquire()
    try:
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    finally:
        if lock:
            lock.release()


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dir_file_count(path):
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
    return n


def dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


def disk_snapshot():
    snap = {}
    for label, path in [('tmp', '/root/autodl-tmp'), ('fs', '/autodl-fs/data')]:
        usage = shutil.disk_usage(path)
        st = os.statvfs(path)
        snap[label] = {
            'path': path,
            'total_bytes': usage.total,
            'used_bytes': usage.used,
            'free_bytes': usage.free,
            'inode_total': st.f_files,
            'inode_free': st.f_ffree,
            'inode_used': st.f_files - st.f_ffree,
        }
    return snap


def check_disk_or_raise(args):
    snap = disk_snapshot()
    tmp_free_gb = snap['tmp']['free_bytes'] / (1024 ** 3)
    fs_free_gb = snap['fs']['free_bytes'] / (1024 ** 3)
    fs_inode_free = snap['fs']['inode_free']
    if tmp_free_gb < args.min_tmp_free_gb:
        raise RuntimeError(f'tmp free space too low: {tmp_free_gb:.1f} GiB < {args.min_tmp_free_gb:.1f} GiB')
    if fs_free_gb < args.min_fs_free_gb:
        raise RuntimeError(f'fs free space too low: {fs_free_gb:.1f} GiB < {args.min_fs_free_gb:.1f} GiB')
    if fs_inode_free < args.min_fs_free_inodes:
        raise RuntimeError(f'fs free inodes too low: {fs_inode_free} < {args.min_fs_free_inodes}')
    return snap


def parse_final_json_payload(text):
    decoder = json.JSONDecoder()
    found = None
    idx = text.find('{')
    while idx != -1:
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except Exception:
            idx = text.find('{', idx + 1)
            continue
        if isinstance(payload, dict) and 'clips' in payload:
            found = payload
        idx = text.find('{', idx + 1)
    if found is not None:
        return found
    raise ValueError('could not find final JSON payload in generator log')


def run_logged(cmd, log_path, cwd=None, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open('w', encoding='utf-8', errors='replace') as log:
        log.write(f'# started {utc_now()}\n')
        log.write('# command redacted: ' + ' '.join(str(x) for x in cmd[:6]) + ' ...\n')
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
        rc = proc.wait()
        log.write(f'\n# finished {utc_now()} rc={rc} elapsed_s={time.time() - started:.1f}\n')
    return rc


def patch_replay_script(group_dir):
    replay = group_dir / 'replay_scene.py'
    if not replay.is_file():
        return
    text = replay.read_text(encoding='utf-8')
    old = "gs.morphs.Mesh(file=obj['visual_mesh']['file'], scale=obj['scale'], pos=st['position_m'], euler=(0,0,obj['yaw_deg']), fixed=True, collision=False, decimate=False, convexify=False, align=False, group_by_material=False)"
    new = "gs.morphs.Mesh(file=str((root / obj['visual_mesh']['file']).resolve()) if not Path(obj['visual_mesh']['file']).is_absolute() else obj['visual_mesh']['file'], scale=obj['scale'], pos=st['position_m'], euler=(0,0,obj['yaw_deg']), fixed=True, collision=False, decimate=False, convexify=False, align=False, group_by_material=False)"
    if old in text:
        replay.write_text(text.replace(old, new), encoding='utf-8')


def update_json_file(path, transform):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    changed = transform(data)
    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return data


def copy_visual_asset(src, batch_dir, asset_cache):
    src_path = Path(src)
    if not src_path.is_file():
        return None
    key = str(src_path.resolve())
    if key in asset_cache:
        return asset_cache[key]
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]
    safe_name = f'{digest}_{src_path.name}'
    dst = batch_dir / 'assets' / 'visual_meshes' / safe_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src_path, dst)
    rel_from_batch = os.path.relpath(dst, batch_dir)
    asset_cache[key] = rel_from_batch
    return rel_from_batch


def make_group_portable(group_dir, batch_dir, old_clip_dir, asset_cache):
    asset_records = []

    def patch_state(data):
        changed = False
        for obj in data.get('objects', []):
            vm = obj.get('visual_mesh') or {}
            cm = obj.get('collision_mesh') or {}
            if 'file' in vm:
                original = vm['file']
                rel_from_batch = copy_visual_asset(original, batch_dir, asset_cache)
                if rel_from_batch:
                    rel_from_group = os.path.relpath(batch_dir / rel_from_batch, group_dir)
                    vm['original_absolute_file'] = original
                    vm['file'] = rel_from_group
                    vm['dataset_relative_file'] = rel_from_batch
                    asset_records.append({'object_id': obj.get('object_id'), 'kind': 'visual_mesh', 'original': original, 'dataset_file_from_group': rel_from_group, 'dataset_file_from_batch': rel_from_batch})
                    changed = True
            if 'file' in cm:
                original = cm['file']
                if str(original).startswith(str(old_clip_dir)):
                    rel = os.path.relpath(original, group_dir)
                    cm['original_absolute_file'] = original
                    cm['file'] = rel
                    cm['dataset_relative_file'] = str(Path('groups') / Path(group_dir).name / rel)
                    changed = True
        data['portable_asset_policy'] = {
            'visual_meshes_copied_to_batch_assets': True,
            'collision_meshes_stored_inside_group': True,
            'paths_are_relative_to_group_for_replay': True,
        }
        return changed or True

    state = update_json_file(group_dir / 'scene_state.json', patch_state)

    def patch_desc(data):
        changed = False
        for obj in data.get('objects', []):
            for key in ('visual_mesh', 'collision_mesh'):
                rec = obj.get(key) or {}
                original = rec.get('file')
                if not original:
                    continue
                if key == 'visual_mesh':
                    rel_from_batch = copy_visual_asset(rec.get('original_absolute_file', original), batch_dir, asset_cache)
                    if rel_from_batch:
                        rel = os.path.relpath(batch_dir / rel_from_batch, group_dir)
                        rec.setdefault('original_absolute_file', original)
                        rec['file'] = rel
                        changed = True
                elif str(original).startswith(str(old_clip_dir)):
                    rec.setdefault('original_absolute_file', original)
                    rec['file'] = os.path.relpath(original, group_dir)
                    changed = True
        return changed

    update_json_file(group_dir / 'descriptions.json', patch_desc)
    patch_replay_script(group_dir)
    (group_dir / 'group_asset_manifest.json').write_text(
        json.dumps({'group': group_dir.name, 'assets': asset_records, 'scene_state': 'scene_state.json'}, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return state or {}


def rewrite_meta_paths(meta, old_clip_dir, group_dir):
    out = {}
    for k, v in meta.items():
        if isinstance(v, str) and v.startswith(str(old_clip_dir)):
            out[k] = os.path.relpath(v, group_dir)
        else:
            out[k] = v
    out['group_dir'] = group_dir.name
    return out


def batch_archive_name(batch_seq, first_gid, last_gid):
    return f'shard_{batch_seq:06d}_groups_{first_gid:08d}_{last_gid:08d}.tar.gz'


class GithubClient:
    def __init__(self, owner, repo, token, private=False):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.private = private
        self.release_id = None
        self.release_upload_url = None

    def request(self, method, url, data=None, headers=None, expected=(200, 201, 204)):
        hdr = {
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {self.token}',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'geniesim-dataset-runner',
        }
        if headers:
            hdr.update(headers)
        body = None
        if data is not None:
            if isinstance(data, (dict, list)):
                body = json.dumps(data).encode('utf-8')
                hdr['Content-Type'] = 'application/json'
            elif isinstance(data, bytes):
                body = data
            else:
                body = data
        req = urllib.request.Request(url, data=body, headers=hdr, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                if resp.status not in expected:
                    raise RuntimeError(f'GitHub {method} {url} returned {resp.status}: {raw[:300]!r}')
                return json.loads(raw.decode('utf-8')) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode('utf-8', 'replace')
            if exc.code in expected:
                return json.loads(raw) if raw else None
            raise RuntimeError(f'GitHub {method} {url} failed {exc.code}: {raw[:1000]}') from exc

    def ensure_repo(self):
        repo_url = f'{GITHUB_API}/repos/{self.owner}/{self.repo}'
        try:
            repo = self.request('GET', repo_url)
            return repo.get('html_url')
        except RuntimeError as exc:
            if 'failed 404' not in str(exc):
                raise
        payload = {
            'name': self.repo,
            'description': 'GenesisWorld + Nyx tabletop SimFoundry input dataset shards and manifests.',
            'private': bool(self.private),
            'has_issues': True,
            'has_projects': False,
            'has_wiki': False,
        }
        repo = self.request('POST', f'{GITHUB_API}/user/repos', payload)
        return repo.get('html_url')

    def put_file(self, path, content, message):
        url_path = urllib.parse.quote(path)
        url = f'{GITHUB_API}/repos/{self.owner}/{self.repo}/contents/{url_path}'
        sha = None
        try:
            current = self.request('GET', url)
            sha = current.get('sha')
        except RuntimeError as exc:
            if 'failed 404' not in str(exc):
                raise
        payload = {
            'message': message,
            'content': base64.b64encode(content.encode('utf-8')).decode('ascii'),
        }
        if sha:
            payload['sha'] = sha
        self.request('PUT', url, payload)

    def ensure_release(self, tag):
        if self.release_id and self.release_upload_url:
            return self.release_id
        url = f'{GITHUB_API}/repos/{self.owner}/{self.repo}/releases/tags/{tag}'
        try:
            rel = self.request('GET', url)
        except RuntimeError as exc:
            if 'failed 404' not in str(exc):
                raise
            rel = self.request('POST', f'{GITHUB_API}/repos/{self.owner}/{self.repo}/releases', {
                'tag_name': tag,
                'name': tag,
                'body': 'Compressed dataset shards. Each shard contains groups, videos, masks, depth previews, state JSON, replay validation, and batch-level copied visual mesh assets.',
                'draft': False,
                'prerelease': False,
            })
        self.release_id = rel['id']
        self.release_upload_url = rel['upload_url'].split('{')[0]
        return self.release_id

    def list_assets(self):
        self.ensure_release('dataset-shards-v1')
        return self.request('GET', f'{GITHUB_API}/repos/{self.owner}/{self.repo}/releases/{self.release_id}/assets') or []

    def delete_asset(self, asset_id):
        self.request('DELETE', f'{GITHUB_API}/repos/{self.owner}/{self.repo}/releases/assets/{asset_id}', expected=(204,))

    def upload_asset(self, path, tag='dataset-shards-v1'):
        if path.stat().st_size > MAX_RELEASE_ASSET_BYTES:
            raise RuntimeError(f'{path.name} is larger than GitHub release asset limit')
        self.ensure_release(tag)
        assets = self.list_assets()
        for asset in assets:
            if asset.get('name') == path.name:
                if int(asset.get('size') or 0) == path.stat().st_size:
                    return asset.get('browser_download_url'), 'already_exists'
                self.delete_asset(asset['id'])
        qs = urllib.parse.urlencode({'name': path.name})
        url = f'{self.release_upload_url}?{qs}'
        headers = {
            'Content-Type': 'application/gzip',
            'Content-Length': str(path.stat().st_size),
        }
        with path.open('rb') as f:
            asset = self.request('POST', url, data=f, headers=headers)
        return asset.get('browser_download_url'), 'uploaded'


def build_readme(args, repo_url=None):
    return f"""# {DATASET_NAME}

GenesisWorld + Nyx tabletop dataset for SimFoundry input testing.

This repository stores scripts, manifests, and GitHub Release links. The actual data is stored as compressed release shards because the dataset contains videos and state sidecars.

Current generation target: {args.total} valid groups.

Each accepted group includes:

- SimFoundry input RGB video without boxes.
- Separate 2D box video and 3D box video.
- Segmentation mask video, depth preview video, per-frame annotations, camera trajectory, object trajectories, scene state, text descriptions, and replay validation.
- Collision proxy meshes inside the group and copied visual mesh assets at the shard level.

Remote backup root:

`{FS_ROOT}`

Generated by:

`{PROJECT_ROOT / 'scripts' / 'run_geniesim_10k_dataset.py'}`
"""


def run_batch(args, batch_seq, requested_count, gpu_id, id_allocator, manifest_lock):
    start_time = time.time()
    snap_before = check_disk_or_raise(args)
    raw_out = TMP_ROOT / 'raw_batches' / f'batch_{batch_seq:06d}_gpu_{gpu_id}'
    batch_dir = TMP_ROOT / 'staging' / f'batch_{batch_seq:06d}'
    groups_dir = batch_dir / 'groups'
    groups_dir.mkdir(parents=True, exist_ok=True)
    if raw_out.exists():
        shutil.rmtree(raw_out)
    raw_out.mkdir(parents=True, exist_ok=True)
    log_path = FS_ROOT / 'logs' / f'batch_{batch_seq:06d}.log'
    seed = args.seed_base + batch_seq * 1000003
    cmd = [
        str(ENV_RUN), 'python', str(GEN_SCRIPT),
        '--renderer', 'nyx',
        '--clips', str(requested_count),
        '--frames', str(args.frames),
        '--width', str(args.width),
        '--height', str(args.height),
        '--fps', str(args.fps),
        '--spp', str(args.spp),
        '--seed', str(seed),
        '--warmup-steps', str(args.warmup_steps),
        '--settle-frames', str(args.settle_frames),
        '--drop-height-min', str(args.drop_height_min),
        '--drop-height-max', str(args.drop_height_max),
        '--min-objects', str(args.min_objects),
        '--max-objects', str(args.max_objects),
        '--asset-source', args.asset_source,
        '--asset-manifest', args.asset_manifest,
        '--cuda-device', str(gpu_id),
        '--out-dir', str(raw_out),
        '--max-attempts', str(args.max_attempts),
    ]
    if args.skip_replay_smoke:
        cmd.append('--skip-replay-smoke')
    rc = run_logged(cmd, log_path, cwd=PROJECT_ROOT)
    text = log_path.read_text(encoding='utf-8', errors='replace')
    if rc != 0:
        raise RuntimeError(f'batch {batch_seq} generator failed rc={rc}; see {log_path}')
    payload = parse_final_json_payload(text)
    passed = []
    failed = []
    asset_cache = {}
    for meta in payload.get('clips', []):
        validation = meta.get('validation') or {}
        if not validation.get('physics_self_consistent'):
            failed.append(meta)
            continue
        old_clip = Path(meta['clip_dir'])
        if not old_clip.is_dir():
            failed.append({**meta, 'runner_error': 'clip_dir missing'})
            continue
        group_id = id_allocator()
        group_name = f'group_{group_id:08d}'
        group_dir = groups_dir / group_name
        shutil.move(str(old_clip), str(group_dir))
        state = make_group_portable(group_dir, batch_dir, old_clip, asset_cache)
        portable_meta = rewrite_meta_paths(meta, old_clip, group_dir)
        portable_meta.update({
            'group_id': group_id,
            'group_name': group_name,
            'batch_seq': batch_seq,
            'archive_name': None,
            'object_categories': [o.get('category') for o in state.get('objects', []) if o.get('object_id') != 'table_000'],
        })
        (group_dir / 'group_manifest.json').write_text(json.dumps(portable_meta, indent=2, ensure_ascii=False), encoding='utf-8')
        passed.append(portable_meta)
    asset_manifest = {
        'batch_seq': batch_seq,
        'visual_mesh_assets': [{'original_absolute_file': k, 'dataset_file_from_batch': v} for k, v in sorted(asset_cache.items())],
    }
    (batch_dir / 'asset_pack_manifest.json').write_text(json.dumps(asset_manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (batch_dir / 'batch_manifest.json').write_text(json.dumps({
        'dataset_name': DATASET_NAME,
        'batch_seq': batch_seq,
        'requested_count': requested_count,
        'accepted_count': len(passed),
        'failed_count': len(failed),
        'groups': passed,
        'failed': failed,
        'disk_before': snap_before,
        'created_at_utc': utc_now(),
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    if not passed:
        if raw_out.exists() and not args.keep_raw:
            shutil.rmtree(raw_out, ignore_errors=True)
        shutil.rmtree(batch_dir, ignore_errors=True)
        return {
            'batch_seq': batch_seq,
            'requested_count': requested_count,
            'accepted_count': 0,
            'failed_count': len(failed),
            'elapsed_s': time.time() - start_time,
            'log': str(log_path),
        }

    first_gid = min(g['group_id'] for g in passed)
    last_gid = max(g['group_id'] for g in passed)
    archive = FS_ROOT / 'archives' / batch_archive_name(batch_seq, first_gid, last_gid)
    tar_cmd = ['tar', '-czf', str(archive), '-C', str(batch_dir.parent), batch_dir.name]
    tar_log = FS_ROOT / 'logs' / f'batch_{batch_seq:06d}_tar.log'
    tar_rc = run_logged(tar_cmd, tar_log)
    if tar_rc != 0 or not archive.is_file():
        raise RuntimeError(f'batch {batch_seq} tar failed rc={tar_rc}; see {tar_log}')
    archive_size = archive.stat().st_size
    if archive_size > MAX_RELEASE_ASSET_BYTES:
        raise RuntimeError(f'{archive} is {archive_size} bytes; reduce --batch-size below GitHub release limit')
    archive_sha256 = sha256_file(archive)
    for g in passed:
        g['archive_name'] = archive.name
        g['archive_sha256'] = archive_sha256
    batch_record = {
        'dataset_name': DATASET_NAME,
        'batch_seq': batch_seq,
        'archive': str(archive),
        'archive_name': archive.name,
        'archive_size_bytes': archive_size,
        'archive_sha256': archive_sha256,
        'accepted_count': len(passed),
        'failed_count': len(failed),
        'groups_first': first_gid,
        'groups_last': last_gid,
        'group_file_count': dir_file_count(batch_dir),
        'expanded_size_bytes': dir_size_bytes(batch_dir),
        'elapsed_s': time.time() - start_time,
        'disk_after': disk_snapshot(),
        'log': str(log_path),
        'tar_log': str(tar_log),
        'created_at_utc': utc_now(),
        'groups': passed,
    }
    append_jsonl(FS_ROOT / 'manifests' / 'global_manifest.jsonl', batch_record, manifest_lock)
    if raw_out.exists() and not args.keep_raw:
        shutil.rmtree(raw_out, ignore_errors=True)
    if not args.keep_staging:
        shutil.rmtree(batch_dir, ignore_errors=True)
    return batch_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=10000)
    ap.add_argument('--batch-size', type=int, default=25)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--gpu-list', default='0')
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--width', type=int, default=480)
    ap.add_argument('--height', type=int, default=270)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--spp', type=int, default=8)
    ap.add_argument('--warmup-steps', type=int, default=4)
    ap.add_argument('--settle-frames', type=int, default=28)
    ap.add_argument('--drop-height-min', type=float, default=0.10)
    ap.add_argument('--drop-height-max', type=float, default=0.24)
    ap.add_argument('--min-objects', type=int, default=5)
    ap.add_argument('--max-objects', type=int, default=7)
    ap.add_argument('--max-attempts', type=int, default=4)
    ap.add_argument('--asset-source', default='geniesim')
    ap.add_argument('--asset-manifest', default='/root/autodl-tmp/mingyu/genesisworld/assets/geniesim3_lfs_glb_tabletop/manifest.json')
    ap.add_argument('--seed-base', type=int, default=2026071300)
    ap.add_argument('--skip-replay-smoke', action='store_true')
    ap.add_argument('--keep-raw', action='store_true')
    ap.add_argument('--keep-staging', action='store_true')
    ap.add_argument('--min-tmp-free-gb', type=float, default=60.0)
    ap.add_argument('--min-fs-free-gb', type=float, default=120.0)
    ap.add_argument('--min-fs-free-inodes', type=int, default=10000)
    ap.add_argument('--github-owner', default='x12979937')
    ap.add_argument('--github-repo', default=DATASET_NAME)
    ap.add_argument('--github-token-env', default='GITHUB_TOKEN')
    ap.add_argument('--github-private', action='store_true')
    ap.add_argument('--enable-github-upload', action='store_true')
    ap.add_argument('--release-tag', default='dataset-shards-v1')
    ap.add_argument('--max-consecutive-failed-batches', type=int, default=5)
    args = ap.parse_args()

    ensure_dirs(args)
    status_path = FS_ROOT / 'status.json'
    manifest_lock = threading.Lock()
    id_lock = threading.Lock()
    upload_lock = threading.Lock()
    batch_lock = threading.Lock()
    accepted_count = 0
    next_group_id = 1
    next_batch_seq = 1
    existing_manifest = FS_ROOT / 'manifests' / 'global_manifest.jsonl'
    if existing_manifest.is_file():
        with existing_manifest.open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                accepted_count += int(rec.get('accepted_count') or 0)
                next_group_id = max(next_group_id, int(rec.get('groups_last') or 0) + 1)
                next_batch_seq = max(next_batch_seq, int(rec.get('batch_seq') or 0) + 1)

    repo_url = None
    gh = None
    if args.enable_github_upload:
        token = os.environ.get(args.github_token_env)
        if not token:
            raise RuntimeError(f'{args.github_token_env} is not set')
        gh = GithubClient(args.github_owner, args.github_repo, token, private=args.github_private)
        repo_url = gh.ensure_repo()
        gh.put_file('README.md', build_readme(args, repo_url), 'Update dataset README')
        gh.put_file('scripts/run_geniesim_10k_dataset.py', Path(__file__).read_text(encoding='utf-8'), 'Add dataset runner')

    def allocate_group_id():
        nonlocal next_group_id
        with id_lock:
            gid = next_group_id
            next_group_id += 1
            return gid

    def allocate_batch_seq():
        nonlocal next_batch_seq
        with batch_lock:
            seq = next_batch_seq
            next_batch_seq += 1
            return seq

    def write_status(extra=None):
        payload = {
            'dataset_name': DATASET_NAME,
            'target_total': args.total,
            'accepted_count': accepted_count,
            'next_group_id': next_group_id,
            'next_batch_seq': next_batch_seq,
            'tmp_root': str(TMP_ROOT),
            'fs_root': str(FS_ROOT),
            'repo_url': repo_url,
            'github_upload_enabled': bool(args.enable_github_upload),
            'updated_at_utc': utc_now(),
            'disk': disk_snapshot(),
        }
        if extra:
            payload.update(extra)
        write_json_atomic(status_path, payload)

    write_status({'state': 'starting'})
    gpus = [x.strip() for x in args.gpu_list.split(',') if x.strip()]
    if not gpus:
        raise RuntimeError('--gpu-list is empty')
    workers = max(1, min(args.workers, len(gpus)))
    in_flight = {}
    in_flight_requested = 0
    completed_batches = 0
    failed_batches = 0
    consecutive_failed_batches = 0
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while accepted_count < args.total or in_flight:
            while accepted_count + in_flight_requested < args.total and len(in_flight) < workers:
                requested = min(args.batch_size, args.total - accepted_count - in_flight_requested)
                if requested <= 0:
                    break
                seq = allocate_batch_seq()
                gpu = gpus[(seq - 1) % len(gpus)]
                fut = pool.submit(run_batch, args, seq, requested, gpu, allocate_group_id, manifest_lock)
                in_flight[fut] = {'seq': seq, 'requested': requested, 'gpu': gpu}
                in_flight_requested += requested
                write_status({'state': 'running', 'in_flight': list(in_flight.values())})
            if not in_flight:
                break
            done, _pending = concurrent.futures.wait(in_flight, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                write_status({'state': 'running', 'in_flight': list(in_flight.values()), 'elapsed_s': time.time() - start})
                continue
            for fut in done:
                info = in_flight.pop(fut)
                in_flight_requested -= info['requested']
                try:
                    rec = fut.result()
                    completed_batches += 1
                    accepted = int(rec.get('accepted_count') or 0)
                    accepted_count += accepted
                    if accepted > 0:
                        consecutive_failed_batches = 0
                    else:
                        consecutive_failed_batches += 1
                    upload_result = None
                    if gh and accepted and rec.get('archive'):
                        with upload_lock:
                            archive = Path(rec['archive'])
                            try:
                                url, action = gh.upload_asset(archive, tag=args.release_tag)
                                upload_result = {'archive': archive.name, 'url': url, 'action': action, 'uploaded_at_utc': utc_now()}
                                append_jsonl(FS_ROOT / 'github' / 'uploads.jsonl', upload_result, manifest_lock)
                            except Exception as upload_exc:
                                upload_result = {'archive': archive.name, 'action': 'upload_failed', 'error': repr(upload_exc), 'uploaded_at_utc': utc_now()}
                                append_jsonl(FS_ROOT / 'github' / 'upload_errors.jsonl', upload_result, manifest_lock)
                    append_jsonl(FS_ROOT / 'manifests' / 'batch_results.jsonl', {**rec, 'github_upload': upload_result}, manifest_lock)
                    write_status({'state': 'running', 'last_completed_batch': rec.get('batch_seq'), 'completed_batches': completed_batches, 'failed_batches': failed_batches, 'in_flight': list(in_flight.values())})
                    if consecutive_failed_batches >= args.max_consecutive_failed_batches:
                        write_status({'state': 'aborted', 'abort_reason': f'{consecutive_failed_batches} consecutive zero-accepted batches'})
                        raise RuntimeError(f'aborting after {consecutive_failed_batches} consecutive zero-accepted batches')
                except Exception as exc:
                    failed_batches += 1
                    consecutive_failed_batches += 1
                    err = {'batch': info, 'error': repr(exc), 'time_utc': utc_now()}
                    append_jsonl(FS_ROOT / 'logs' / 'batch_errors.jsonl', err, manifest_lock)
                    write_status({'state': 'running_with_errors', 'last_error': err, 'completed_batches': completed_batches, 'failed_batches': failed_batches, 'in_flight': list(in_flight.values())})
                    if consecutive_failed_batches >= args.max_consecutive_failed_batches:
                        write_status({'state': 'aborted', 'abort_reason': f'{consecutive_failed_batches} consecutive failed batches', 'last_error': err})
                        raise RuntimeError(f'aborting after {consecutive_failed_batches} consecutive failed batches') from exc
                    time.sleep(5)

    final_status = {
        'state': 'complete' if accepted_count >= args.total else 'stopped',
        'accepted_count': accepted_count,
        'completed_batches': completed_batches,
        'failed_batches': failed_batches,
        'elapsed_s': time.time() - start,
    }
    if gh:
        gh.put_file('manifests/status.json', status_path.read_text(encoding='utf-8'), 'Update dataset status')
        if (FS_ROOT / 'github' / 'uploads.jsonl').is_file():
            gh.put_file('manifests/uploads.jsonl', (FS_ROOT / 'github' / 'uploads.jsonl').read_text(encoding='utf-8'), 'Update upload manifest')
        if (FS_ROOT / 'github' / 'upload_errors.jsonl').is_file():
            gh.put_file('manifests/upload_errors.jsonl', (FS_ROOT / 'github' / 'upload_errors.jsonl').read_text(encoding='utf-8'), 'Update upload error manifest')
        gh.put_file('manifests/global_manifest.head.jsonl', ''.join((FS_ROOT / 'manifests' / 'global_manifest.jsonl').read_text(encoding='utf-8').splitlines(True)[-200:]), 'Update manifest tail')
    write_status(final_status)
    print(json.dumps(final_status, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
