"""Bounded live timing delivery; no network or inference work on producer threads."""
import asyncio
from collections import deque
import json
from threading import Lock
from datetime import datetime, timezone


MAX_EVENT_BYTES = 1_048_576
MAX_SUBSCRIBERS = 4
MAX_PENDING = 8


class TimingSubscription:
    def __init__(self, loop):
        self.loop, self.event, self.lock = loop, asyncio.Event(), Lock()
        self.pending, self.closed, self.notified = deque(), False, False

    def offer(self, frame, gap, terminal=False):
        with self.lock:
            if self.closed:
                return
            if len(self.pending) >= MAX_PENDING:
                self.pending.clear()
                frame, terminal = gap, True
            self.pending.append(frame)
            self.closed = terminal
            if not self.notified:
                self.notified = True
                try:
                    self.loop.call_soon_threadsafe(self.event.set)
                except RuntimeError:
                    self.pending.clear()
                    self.closed = True

    async def next(self):
        while True:
            with self.lock:
                if self.pending:
                    return self.pending.popleft()
                if self.closed:
                    return None
                self.notified = False
                self.event.clear()
            await self.event.wait()

    def close(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
        try:
            self.loop.call_soon_threadsafe(self.event.set)
        except RuntimeError:
            pass


class TimingStreamHub:
    def __init__(self, source_id, resource):
        self.source_id, self.resource = source_id, dict(resource)
        self.subscribers, self.lock = set(), Lock()

    def identity(self, cursor, kind):
        return dict(version=1, type=kind, sourceId=self.source_id, instanceId=self.resource['instanceId'],
                    resource=self.resource, cursor=str(cursor), observedAt=datetime.now(timezone.utc).isoformat())

    def frame(self, cursor, kind, **values):
        return 'data: ' + json.dumps({**self.identity(cursor, kind), **values}, allow_nan=False) + '\n\n'

    def subscribe(self, loop, cursor):
        with self.lock:
            if len(self.subscribers) >= MAX_SUBSCRIBERS:
                return None
            subscriber = TimingSubscription(loop)
            ready = self.frame(cursor, 'ready')
            subscriber.offer(ready, ready)
            self.subscribers.add(subscriber)
            return subscriber

    def remove(self, subscriber):
        with self.lock:
            self.subscribers.discard(subscriber)
        subscriber.close()

    def publish(self, cursor, record_json):
        with self.lock:
            subscribers = tuple(subscriber for subscriber in self.subscribers if not subscriber.closed)
        if not subscribers:
            return
        # Both JSON strings use ensure_ascii=True, so character count equals bytes.
        payload = None
        if len(record_json) <= MAX_EVENT_BYTES:
            payload = json.dumps(self.identity(cursor, 'observation'))[:-1] + ', "observation": ' + record_json + '}'
        oversized = payload is None or len(payload) > MAX_EVENT_BYTES
        gap = self.frame(cursor, 'gap', reasons=['timing_stream_event_too_large' if oversized else 'timing_stream_overflow'])
        frame = gap if oversized else 'data: ' + payload + '\n\n'
        for subscriber in subscribers:
            subscriber.offer(frame, gap, terminal=oversized)
