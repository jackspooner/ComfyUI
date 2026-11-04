"""Read-only local operator boundary for compact timing facts."""
import ipaddress
from urllib.parse import urlsplit
from aiohttp import web


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
        except ValueError:
            return web.json_response({'error': 'timing_invalid_cursor'}, status=400)
        except Exception:
            return web.json_response({'error': 'timing_unavailable'}, status=503)
