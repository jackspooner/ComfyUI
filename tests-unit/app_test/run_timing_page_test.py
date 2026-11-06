import asyncio
import json
from threading import RLock
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from app import run_timing
from app.run_timing_http import add_routes
from app_test.run_timing_stream_test import item


def test_byte_bounded_pages_preserve_every_event_cursor_and_read_only_queue_snapshot(tmp_path, monkeypatch):
    store = run_timing.RunTimingStore(str(tmp_path / 'timing.sqlite'))
    store.event('accepted', item())
    store.event('started', item())
    for _ in range(30):
        store.safe_stage('prompt', 'node', 'metrics')
    before = store.db.total_changes
    all_records = [(seq, json.loads(raw)) for seq, raw in store.db.execute('SELECT seq,record FROM timing_events ORDER BY seq')]
    active = store.page('0', 1, [item()], [])['active']
    monkeypatch.setattr(run_timing, 'MAX_TIMING_PAGE_BYTES', 12_000)
    cursor, received, pages = '0', [], []
    while True:
        page = store.page(cursor, 500, [item()], [])
        assert len(json.dumps(page).encode('utf8')) <= 12_000
        assert page['active']['complete'] is True
        assert [r['id'] for r in page['active']['records']] == [r['id'] for r in active['records']]
        assert [r['queueOrder'] for r in page['active']['records']] == [0]
        received.extend(page['observations'])
        pages.append(page)
        if page['nextCursor'] is None:
            break
        assert int(page['cursor']) > int(cursor)
        cursor = page['nextCursor']
    assert len(pages) > 1
    assert received == [record for _, record in all_records]
    assert page['cursor'] == str(all_records[-1][0])
    assert store.db.total_changes == before


@pytest.mark.parametrize('active', [False, True])
def test_single_oversized_record_or_snapshot_never_advances_or_silently_truncates(tmp_path, monkeypatch, active):
    store = run_timing.RunTimingStore(str(tmp_path / 'timing.sqlite'))
    store.event('accepted', item())
    before = store.db.total_changes
    monkeypatch.setattr(run_timing, 'MAX_TIMING_PAGE_BYTES', 1000)
    with pytest.raises(run_timing.TimingPageTooLarge):
        store.page('0', 500, [item()] if active else [], [])
    assert store.db.total_changes == before
    assert store.db.execute('SELECT count(*) FROM timing_events').fetchone()[0] == 1


def test_oversize_http_response_is_explicit_and_read_only(tmp_path, monkeypatch):
    async def scenario():
        store = run_timing.RunTimingStore(str(tmp_path / 'timing.sqlite'))
        store.event('accepted', item())
        queue = SimpleNamespace(timing=store, mutex=RLock(), get_current_queue_volatile=lambda: ([item()], []))
        routes = web.RouteTableDef()
        add_routes(routes, queue)
        app = web.Application()
        app.add_routes(routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            before = store.db.total_changes
            monkeypatch.setattr(run_timing, 'MAX_TIMING_PAGE_BYTES', 1000)
            response = await client.get('/timing/observations?after=0&limit=500')
            assert response.status == 503 and await response.json() == {'error': 'timing_page_too_large'}
            assert store.db.total_changes == before
        finally:
            await client.close()
            store.db.close()
    asyncio.run(scenario())
