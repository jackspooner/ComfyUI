"""Read-only local operator boundary for compact timing facts."""
import ipaddress
import asyncio
from urllib.parse import urlsplit
from aiohttp import web
from app.run_timing import TimingPageTooLarge


def local_request(request):
    if any(key in request.headers for key in ('Origin', 'Sec-Fetch-Site', 'Forwarded', 'X-Forwarded-For', 'X-Forwarded-Host')):
        return False
    try:
        remote = ipaddress.ip_address(request.remote)
        host = urlsplit('//' + request.host).hostname
        return remote.is_loopback and (host == 'localhost' or ipaddress.ip_address(host).is_loopback)
    except (ValueError, TypeError):
        return False


def add_routes(routes, queue):
    @routes.get('/timing/observations/stream')
    async def stream(request):
        if not local_request(request):
            return web.json_response({'error': 'timing_local_only'}, status=403)
        if queue.timing is None:
            return web.json_response({'error': 'timing_unavailable'}, status=503)
        try:
            with queue.mutex:
                cursor = queue.timing.db.execute('SELECT coalesce(max(seq),0) FROM timing_events').fetchone()[0]
                hub = queue.timing.streams
                subscriber = hub.subscribe(asyncio.get_running_loop(), cursor)
        except Exception:
            return web.json_response({'error': 'timing_unavailable'}, status=503)
        if subscriber is None:
            return web.json_response({'error': 'timing_stream_limit'}, status=429)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
        try:
            await response.prepare(request)
            while True:
                try:
                    frame = await asyncio.wait_for(subscriber.next(), timeout=15)
                except TimeoutError:
                    frame = ': heartbeat\n\n'
                if frame is None:
                    break
                await asyncio.wait_for(response.write(frame.encode('utf8')), timeout=5)
        except (ConnectionError, TimeoutError):
            pass
        finally:
            hub.remove(subscriber)
        return response

    @routes.get('/timing/observations')
    async def observations(request):
        if not local_request(request):
            return web.json_response({'error': 'timing_local_only'}, status=403)
        if queue.timing is None:
            return web.json_response({'error': 'timing_unavailable'}, status=503)
        try:
            with queue.mutex:
                running, pending = queue.get_current_queue_volatile()
                page = queue.timing.page(request.query.get('after', '0'), int(request.query.get('limit', '100')), running, pending)
            return web.json_response(page, headers={'Cache-Control': 'no-store'})
        except TimingPageTooLarge:
            return web.json_response({'error': 'timing_page_too_large'}, status=503, headers={'Cache-Control': 'no-store'})
        except ValueError:
            return web.json_response({'error': 'timing_invalid_cursor'}, status=400)
        except Exception:
            return web.json_response({'error': 'timing_unavailable'}, status=503)
