"""Durable pending-only holds, serialized with the native dequeue mutex."""
import heapq
import json
import os
from pathlib import Path
import sqlite3
import uuid
from filelock import FileLock


class QueueControlError(Exception):
    def __init__(self, code, blocked=None):
        super().__init__(code)
        self.code = code
        self.blocked = blocked or []


class QueueControl:
    def __init__(self, filename):
        directory = Path(filename).parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = FileLock(str(filename) + '.lock')
        self.lock.acquire(timeout=0)
        self.db = sqlite3.connect(filename, check_same_thread=False)
        os.chmod(filename, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS control_state (singleton INTEGER PRIMARY KEY, revision INTEGER NOT NULL);
            INSERT OR IGNORE INTO control_state VALUES (1,0);
            CREATE TABLE IF NOT EXISTS held_prompts (id TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS control_operations (id TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL);
        ''')
        if 'revision' not in {row[1] for row in self.db.execute('PRAGMA table_info(held_prompts)')}:
            self.db.execute('ALTER TABLE held_prompts ADD COLUMN revision INTEGER NOT NULL DEFAULT 0')
            self.db.commit()

    def close(self):
        self.db.close()
        self.lock.release()

    def snapshot(self):
        return {'version': 1, 'revision': self.revision(), 'total': self.db.execute('SELECT count(*) FROM held_prompts').fetchone()[0], 'prompts': [
            {'promptId': row['id'], 'state': row['state']} for row in
            self.db.execute('SELECT id,state FROM held_prompts ORDER BY id LIMIT 200')]}

    def revision(self):
        return self.db.execute('SELECT revision FROM control_state').fetchone()[0]

    def held(self):
        return [tuple(json.loads(row[0])) for row in self.db.execute("SELECT payload FROM held_prompts WHERE state='paused'")]


class QueueControlMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_control_store = None

    def configure_queue_control(self, filename):
        self.queue_control_store = QueueControl(filename)
        for item in self.queue_control_store.held():
            self.restore_held_receipt(item)

    def restore_held_receipt(self, item):
        # Plain queues have no maintenance receipt owner.
        pass

    def check_control_resume(self):
        pass

    def put(self, item):
        with self.mutex:
            if self.queue_control_store and self.queue_control_store.db.execute('SELECT 1 FROM held_prompts WHERE id=?', (item[1],)).fetchone():
                # An old producer must not enqueue another executable copy of a
                # held (or uncertain resumed) native identity.
                raise QueueControlError('queue_control_identity_retained')
            return super().put(item)

    def queue_control_snapshot(self):
        with self.mutex:
            return self.queue_control_store.snapshot()

    def control_pending(self, request):
        with self.mutex:
            return self._control_pending(request)

    def _control_pending(self, request):
        store = self.queue_control_store
        db = store.db
        encoded = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False)
        old = db.execute('SELECT request,response FROM control_operations WHERE id=?', (request['operationId'],)).fetchone()
        if old:
            response = json.loads(old['response'])
            current = [db.execute('SELECT revision FROM held_prompts WHERE id=?', (key,)).fetchone() for key in request['promptIds']]
            if old['request'] != encoded or any(not row or row['revision'] != response['revision'] for row in current):
                raise QueueControlError('queue_control_revision_conflict')
            active = {item[1] for item in [*self.queue, *self.currently_running.values()]}
            if response['state'] == 'paused' and any(key in active or self.get_history(key) for key in request['promptIds']):
                raise QueueControlError('queue_control_uncertain')
            if response['state'] == 'resuming':
                if not all(key in active or self.get_history(key) for key in request['promptIds']):
                    raise QueueControlError('queue_control_uncertain')
                response['state'] = 'resumed'
                with db:
                    db.executemany("UPDATE held_prompts SET state='released' WHERE id=?", [(key,) for key in request['promptIds']])
                    db.execute('UPDATE control_operations SET response=? WHERE id=?', (json.dumps(response), request['operationId']))
            return response
        if request['expectedRevision'] != store.revision():
            raise QueueControlError('queue_control_revision_conflict')
        pending = {item[1]: item for item in self.queue}
        running = {item[1] for item in self.currently_running.values()}
        retained = {row['id']: row for row in db.execute('SELECT * FROM held_prompts')}
        selected, blocked = {}, []
        for prompt_id in request['promptIds']:
            row = retained.get(prompt_id)
            if sum(item[1] == prompt_id for item in self.queue) > 1 or prompt_id in pending and self.get_history(prompt_id):
                blocked.append({'promptId': prompt_id, 'reason': 'identity_collision'})
            elif prompt_id in running:
                blocked.append({'promptId': prompt_id, 'reason': 'running'})
            elif request['action'] == 'pause' and row and row['state'] == 'paused' and prompt_id not in pending:
                selected[prompt_id] = tuple(json.loads(row['payload']))
            elif request['action'] == 'pause' and prompt_id in pending and (not row or row['state'] == 'released'):
                selected[prompt_id] = pending[prompt_id]
            elif request['action'] == 'resume' and row and row['state'] == 'paused' and prompt_id not in pending:
                selected[prompt_id] = tuple(json.loads(row['payload']))
            else:
                blocked.append({'promptId': prompt_id, 'reason': 'terminal' if self.get_history(prompt_id) else 'unknown'})
        if blocked:
            raise QueueControlError('queue_control_conflict', blocked)
        if request['action'] == 'resume':
            self.check_control_resume()
        revision = store.revision() + 1
        response = {'version': 1, 'operationId': request['operationId'], 'revision': revision,
                    'action': request['action'], 'state': 'paused' if request['action'] == 'pause' else 'resumed',
                    'promptIds': request['promptIds'], 'receiptId': str(uuid.uuid4())}
        # Commit the safety intent before altering the heap. A paused row is
        # executable only through this explicit resume, never through put().
        with db:
            for prompt_id, item in selected.items():
                db.execute('INSERT INTO held_prompts VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,payload=excluded.payload,revision=excluded.revision',
                           (prompt_id, 'paused' if request['action'] == 'pause' else 'resuming', json.dumps(item, allow_nan=False), revision))
            db.execute('UPDATE control_state SET revision=?', (revision,))
            db.execute('INSERT INTO control_operations VALUES (?,?,?)', (request['operationId'], encoded,
                       json.dumps({**response, 'state': 'resuming'} if request['action'] == 'resume' else response)))
        if request['action'] == 'pause':
            self.queue = [item for item in self.queue if item[1] not in selected]
            heapq.heapify(self.queue)
        else:
            # Do not call put(): acceptance and timing already belong to the
            # original prompt. A crash here retains resuming/unknown, not replay.
            for item in selected.values():
                heapq.heappush(self.queue, item)
            with db:
                db.executemany("UPDATE held_prompts SET state='released' WHERE id=?", [(key,) for key in selected])
                db.execute('UPDATE control_operations SET response=? WHERE id=?', (json.dumps(response), request['operationId']))
            self.not_empty.notify_all()
        self.server.queue_updated()
        return response
