"""Local, prompt-free execution observations. This module never schedules work."""
import hashlib
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import logging
import math
import os
import re
import socket
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def runtime_identity():
    try:
        revision = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = 'unknown'
    try:
        torch_version = version('torch')
    except PackageNotFoundError:
        torch_version = 'unknown'
    return revision, torch_version


FEATURES = {'width', 'height', 'steps', 'num_frames', 'frames', 'frame_count', 'length', 'fps',
            'duration', 'duration_seconds', 'batch_size', 'max_new_tokens', 'max_tokens', 'max_length', 'max_output_tokens', 'context_length', 'context_budget', 'megapixels', 'denoise'}
SETTINGS = {'precision', 'dtype', 'weight_dtype', 'sampler_name', 'scheduler', 'attention', 'sampling_profile', 'decoder'}
MODELS = {'model', 'model_id', 'ckpt_name', 'unet_name', 'diffusion_model', 'model_name', 'clip_name', 'clip_name1', 'clip_name2',
          'clip_name3', 'vae_name', 'checkpoint', 'checkpoint_name', 'lora_name', 'text_encoder'}
WORKLOAD_RESOLVERS = []


def selected_file_identity(key, value):
    """Use the native path registry; keep file identity facts hashed and prompt-free."""
    categories = {'ckpt_name': 'checkpoints', 'checkpoint_name': 'checkpoints', 'unet_name': 'diffusion_models',
                  'diffusion_model': 'diffusion_models', 'vae_name': 'vae', 'lora_name': 'loras',
                  'clip_name': 'text_encoders', 'clip_name1': 'text_encoders', 'clip_name2': 'text_encoders', 'clip_name3': 'text_encoders'}
    category = categories.get(key)
    if not category:
        return None
    try:
        import folder_paths
        filename = folder_paths.get_full_path(category, value)
        if filename:
            facts = Path(filename).stat()
            return digest([facts.st_dev, facts.st_ino, facts.st_size, facts.st_mtime_ns, facts.st_ctime_ns])
    except (OSError, KeyError, ValueError):
        pass
    return None


def register_workload_resolver(resolver):
    """Extensions may describe their immutable graph expansions without executing them."""
    if resolver not in WORKLOAD_RESOLVERS:
        WORKLOAD_RESOLVERS.append(resolver)


def report_execution_features(features):
    """Allow native nodes to add numeric evidence; never let telemetry fail a node."""
    try:
        from comfy_execution.utils import get_executing_context
        from server import PromptServer
        context = get_executing_context()
        queue = PromptServer.instance.prompt_queue
        if context and queue.timing:
            with queue.mutex:
                queue.timing.numeric_evidence(context.prompt_id, features)
    except Exception:
        logging.debug('Native timing feature evidence unavailable')


def physical_gpu_uuid():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=3)
        selected = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
        for line in result.stdout.splitlines():
            index, identity = (part.strip() for part in line.split(','))
            if selected in (index, identity):
                return identity
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def workload(graph):
    """Only explicit execution controls/weight identities enter the timing record."""
    revision, torch_version = runtime_identity()
    components, features, settings, structure = [], {}, {'nativeRevision': revision, 'torchVersion': torch_version, 'artifactAssurance': 'selection'}, []
    signatures = {}
    selected_files = []
    def signature(node_id, visited):
        if node_id in signatures:
            return signatures[node_id]
        if node_id in visited or len(visited) > 64:
            raise ValueError('Timing graph structure is cyclic or too deep')
        node = graph[node_id]
        links = []
        for key, value in node.get('inputs', {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph:
                links.append([key, signature(value[0], visited | {node_id}), value[1]])
        signatures[node_id] = digest([node.get('class_type'), sorted(node.get('inputs', {})), sorted(links)])
        return signatures[node_id]
    structure = sorted(signature(node_id, set()) for node_id in graph)
    for node in graph.values():
        kind, inputs = node.get('class_type', ''), node.get('inputs', {})
        features[kind + '.nodeInstances'] = features.get(kind + '.nodeInstances', 0) + 1
        for key, value in inputs.items():
            if key in MODELS and isinstance(value, str) and value:
                # Hash the owner-selected artifact identity; never store private paths.
                identity = selected_file_identity(key, value)
                selected_files.append(identity is not None)
                components.append(digest([kind, key, value, identity]))
            elif key in FEATURES and type(value) in (float, int) and math.isfinite(value) and value >= 0:
                name = kind + '.' + key
                features[name] = features.get(name, 0) + value
            elif key in ('prompt', 'text', 'messages_json') and isinstance(value, str):
                features[kind + '.inputCharacters'] = features.get(kind + '.inputCharacters', 0) + len(value)
            elif key in SETTINGS and type(value) in (str, bool, int, float):
                settings[kind + '.' + key] = value if not isinstance(value, str) else value[:128]
    components = sorted(set(components))
    if selected_files and all(selected_files):
        settings['artifactAssurance'] = 'selected_file_identity'
    precision = sorted(str(value) for key, value in settings.items() if any(word in key for word in ('precision', 'dtype')))
    return {'model': digest(components) if components else None, 'profileRevision': digest(sorted(set(structure))),
            'components': components, 'executionMode': 'comfyui', 'precision': ','.join(precision) or None,
            'residency': 'unknown', 'cache': 'unknown', 'features': features, 'settings': settings}


class RunTimingStore:
    def __init__(self, filename, host_id=None, gpu_uuid=None, now=time.monotonic):
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        self.db = sqlite3.connect(filename, check_same_thread=False, timeout=1)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS timing_source(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS timing_records(id TEXT PRIMARY KEY, record TEXT NOT NULL, terminal_at REAL);
            CREATE TABLE IF NOT EXISTS timing_events(seq INTEGER PRIMARY KEY AUTOINCREMENT, record TEXT NOT NULL, at REAL NOT NULL);''')
        source = self.db.execute('SELECT id FROM timing_source').fetchone()
        self.source_id = source[0] if source else 'comfyui-' + str(uuid.uuid4())
        if not source:
            self.db.execute('INSERT INTO timing_source VALUES(?)', (self.source_id,))
        self.instance_id = str(uuid.uuid4())
        self.resource = {'hostId': host_id or socket.gethostname(), 'gpuUuid': gpu_uuid,
                         'instanceId': self.instance_id, 'laneId': 'comfyui'}
        self.now, self.clocks, self.issue = now, {}, None
        self.gpu_waits = {}
        self.pruned_at = 0
        self.db.commit()
        # Surviving observations are evidence of interruption, never a replay list.
        for raw, in self.db.execute('SELECT record FROM timing_records WHERE terminal_at IS NULL').fetchall():
            record = json.loads(raw)
            record['phase'] = 'unknown'
            record['coverageReasons'] = ['native_instance_restarted']
            self._save(record, terminal=True)

    def _save(self, record, terminal=False):
        record['revision'] += 1
        record['observedAt'] = utc()
        value, now = json.dumps(record, allow_nan=False), time.time()
        with self.db:
            self.db.execute('INSERT INTO timing_records VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record,terminal_at=excluded.terminal_at',
                            (record['id'], value, now if terminal else None))
            self.db.execute('INSERT INTO timing_events(record,at) VALUES(?,?)', (value, now))
            if now - self.pruned_at > 60:
                self.db.execute('DELETE FROM timing_events WHERE at<?', (now - 90 * 86400,))
                self.db.execute('DELETE FROM timing_records WHERE terminal_at<?', (now - 90 * 86400,))
                self.pruned_at = now
        self.issue = None

    def numeric_evidence(self, prompt_id, features):
        row = self.db.execute('SELECT record FROM timing_records WHERE id=?', (prompt_id,)).fetchone()
        if not row:
            return
        record = json.loads(row[0])
        for key, value in features.items():
            if key in ('inputTokens', 'outputTokens', 'outputAllowance') and type(value) in (int, float) and math.isfinite(value) and value >= 0:
                record['workload']['features']['observed.' + key] = value
        self._save(record)

    def safe_gpu_wait(self, prompt_id, state, elapsed=0):
        """Pause execution accounting during the existing resource owner's wait."""
        try:
            clock = self.clocks.get(prompt_id)
            if not clock or clock[1] is None:
                return
            row = self.db.execute('SELECT record FROM timing_records WHERE id=?', (prompt_id,)).fetchone()
            if not row:
                return
            record = json.loads(row[0])
            wait = self.gpu_waits.setdefault(prompt_id, [0, None])
            if state == 'waiting' and wait[1] is None:
                wait[1] = max(clock[1], self.now() - max(0, float(elapsed)))
                record['coverageReasons'].append('gpu_hold_active')
            elif state == 'cleared' and wait[1] is not None:
                wait[0] += max(0, self.now() - wait[1])
                wait[1] = None
                record['coverageReasons'] = [reason for reason in record['coverageReasons'] if reason != 'gpu_hold_active']
            else:
                return
            self._save(record)
        except Exception:
            self.issue = 'timing_gpu_wait_observation_unavailable'

    def elapsed_phases(self, prompt_id, clock):
        wait, since = self.gpu_waits.get(prompt_id, [0, None])
        held = wait + (max(0, self.now() - since) if since is not None else 0)
        return max(0, self.now() - clock[1] - held), max(0, clock[1] - clock[0]) + held, held

    def resolve_workload(self, record, graph, phase):
        record['coverageReasons'] = [reason for reason in record['coverageReasons'] if reason != 'workload_resolution_unavailable']
        for resolver in WORKLOAD_RESOLVERS:
            try:
                resolved = resolver(graph, phase)
            except Exception:
                record['coverageReasons'].append('workload_resolution_unavailable')
                break
            if resolved is not None:
                observed = {key: value for key, value in record['workload']['features'].items() if key.startswith('observed.')}
                record['workload'] = resolved['workload']
                record['workload']['features'].update(observed)
                record['relatedOperationIds'] = sorted(set(record.get('relatedOperationIds', [])) | set(resolved.get('relatedOperationIds', [])))
                for key in ('batchId', 'stageId', 'sharedOperationId', 'dependsOn'):
                    if key in resolved:
                        record[key] = resolved[key]
                break

    def event(self, action, item, status=None):
        prompt_id = item[1]
        row = self.db.execute('SELECT record FROM timing_records WHERE id=?', (prompt_id,)).fetchone()
        record = json.loads(row[0]) if row else None
        if action == 'accepted':
            if record and record['resource']['instanceId'] == self.instance_id:
                return
            record = {'version': 1, 'id': prompt_id, 'revision': 0, 'owner': 'comfyui', 'operationId': prompt_id,
                      'attemptId': self.instance_id + ':' + prompt_id, 'nativePromptId': prompt_id, 'dependsOn': [],
                      'resource': self.resource, 'workload': workload(item[2]), 'phase': 'queued',
                      'acceptedAt': utc(), 'submittedAt': utc(), 'observedAt': utc(),
                      'durations': {'queueSeconds': None, 'executionSeconds': None, 'publicationSeconds': None},
                      'measurement': 'native_monotonic', 'queueOrder': None, 'orderKnown': True, 'coverageReasons': []}
            identity_pattern = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')
            candidates = [value for node in item[2].values() for key, value in node.get('inputs', {}).items()
                          if key in ('request_id', 'invocation_id', 'run_id', 'batch_id', 'child_id') and isinstance(value, str)]
            if isinstance(item[3].get('request_id'), str):
                candidates.append(item[3]['request_id'])
            for name in ('routervideo', 'workspace2_timing', 'mediatools_tracking', 'mediatools_ocr'):
                meta = item[3].get(name, {})
                if isinstance(meta, dict):
                    candidates.extend(value for key, value in meta.items() if key in ('runId', 'requestId', 'operationId', 'jobId', 'batchId', 'promptId') and isinstance(value, str))
            record['relatedOperationIds'] = sorted({value for value in candidates if len(value) <= 128 and identity_pattern.fullmatch(value)})
            self.resolve_workload(record, item[2], action)
            self.clocks[prompt_id] = [self.now(), None, len(item[2])]
        elif record is None:
            self.issue = 'timing_acceptance_missing'
            return
        elif record['phase'] in ('succeeded', 'failed', 'cancelled'):
            return
        elif action == 'started':
            record.update(phase='running', startedAt=utc())
            clock = self.clocks.get(prompt_id)
            if clock:
                clock[1] = self.now()
                record['durations']['queueSeconds'] = max(0, clock[1] - clock[0])
        else:
            if 'workload_resolution_unavailable' in record['coverageReasons']:
                self.resolve_workload(record, item[2], action)
            clock = self.clocks.pop(prompt_id, None)
            record.update(phase='cancelled' if action == 'cancelled' else 'succeeded' if status and status.completed else 'failed', finishedAt=utc())
            if clock and clock[1] is not None:
                execution, queued, held = self.elapsed_phases(prompt_id, clock)
                record['durations'].update(executionSeconds=execution, queueSeconds=queued)
                record['workload']['features']['observed.gpuHoldWaitSeconds'] = held
            self.gpu_waits.pop(prompt_id, None)
            if record['phase'] == 'succeeded':
                record['readyAt'] = record['finishedAt']
            messages = status.messages if status else []
            cached = [data.get('nodes', []) for event, data in messages if event == 'execution_cached']
            if record['workload']['cache'] != 'unknown':
                pass
            elif cached and len(cached[0]) == len(item[2]):
                record['workload']['cache'] = 'hit'
            elif cached and cached[0]:
                record['workload']['cache'] = 'partial'
            elif action != 'cancelled':
                record['workload']['cache'] = 'none'
        self._save(record, terminal=action not in ('accepted', 'started'))

    def safe_event(self, *args):
        try:
            self.event(*args)
        except Exception:
            if self.issue != 'timing_store_unavailable':
                logging.warning('Run timing observation unavailable; execution continues', exc_info=False)
            self.issue = 'timing_store_unavailable'

    def safe_cache(self, prompt_id, cached_nodes):
        try:
            row = self.db.execute('SELECT record FROM timing_records WHERE id=?', (prompt_id,)).fetchone()
            clock = self.clocks.get(prompt_id)
            if row and clock:
                record = json.loads(row[0])
                if record['workload']['cache'] == 'unknown':
                    record['workload']['cache'] = 'hit' if len(cached_nodes) == clock[2] else 'partial' if cached_nodes else 'none'
                    self._save(record)
        except Exception:
            self.issue = 'timing_cache_observation_unavailable'

    def page(self, cursor, limit, running, pending):
        after = int(cursor or '0')
        if after < 0 or not 1 <= limit <= 500:
            raise ValueError('Invalid timing cursor or limit')
        minimum, maximum = self.db.execute('SELECT min(seq),max(seq) FROM timing_events').fetchone()
        expired = bool(after and ((minimum and after < minimum - 1) or after > (maximum or 0)))
        rows = self.db.execute('SELECT seq,record FROM timing_events WHERE seq>? ORDER BY seq LIMIT ?', (after, limit + 1)).fetchall()
        selected, active = rows[:limit], []
        reasons = [self.issue] if self.issue else []
        for index, item in enumerate(list(running) + sorted(pending, key=lambda value: value[0])):
            row = self.db.execute('SELECT record FROM timing_records WHERE id=?', (item[1],)).fetchone()
            if row:
                record = json.loads(row[0])
                if 'workload_resolution_unavailable' in record['coverageReasons']:
                    self.resolve_workload(record, item[2], 'observation')
                record['queueOrder'] = index
                record['observedAt'] = utc()
                clock = self.clocks.get(item[1])
                if clock:
                    if clock[1] is not None:
                        execution, queued, held = self.elapsed_phases(item[1], clock)
                        record['durations'].update(executionSeconds=execution, queueSeconds=queued)
                        record['workload']['features']['observed.gpuHoldWaitSeconds'] = held
                    else:
                        record['durations']['queueSeconds'] = max(0, self.now() - clock[0])
                active.append(record)
            else:
                reasons.append('timing_acceptance_missing')
        current = str(selected[-1][0] if selected else maximum or 0)
        return {'version': 1, 'resource': self.resource, 'sourceId': self.source_id, 'instanceId': self.instance_id, 'cursor': current,
                'nextCursor': current if len(rows) > limit else None, 'cursorExpired': expired,
                'observedAt': utc(), 'observations': [json.loads(row[1]) for row in selected],
                'active': {'complete': not reasons, 'records': active, 'reasons': sorted(set(reasons))}}
