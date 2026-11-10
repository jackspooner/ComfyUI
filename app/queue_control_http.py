"""Loopback-only exact native pending control. No browser-origin access."""
import asyncio
import json
import logging
import re
import sqlite3
import uuid

from aiohttp import web
from app.maintenance_store import MaintenanceError
from app.queue_control import QueueControlError
from app.run_timing_http import local_request


def validate_request(value):
    if not isinstance(value, dict) or set(value) != {'action', 'operationId', 'expectedRevision', 'promptIds'}:
        raise ValueError('Invalid queue control fields')
    if value['action'] not in ('pause', 'resume') or type(value['expectedRevision']) is not int or not 0 <= value['expectedRevision'] <= 9007199254740991:
        raise ValueError('Invalid queue control action or revision')
    ids = value['promptIds']
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 200
            or any(not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', key) for key in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(value['operationId'], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value['operationId'])):
        raise ValueError('Invalid queue control identities')
    return value


def add_routes(routes, queue):
    @routes.get('/queue/control')
    @routes.post('/queue/control')
    async def control(request):
        request_id = 'req_' + uuid.uuid4().hex
        code, status, blocked = 'queue_control_unavailable', 503, []
        try:
            if not local_request(request):
                code, status = 'queue_control_local_only', 403
            elif request.method == 'GET':
                return web.json_response(queue.queue_control_snapshot())
            else:
                raw = bytearray()
                async for chunk in request.content.iter_chunked(16384):
                    raw.extend(chunk)
                    if len(raw) > 64 * 1024:
                        raise ValueError('Queue control request too large')
                value = validate_request(json.loads(raw))
                result = await asyncio.to_thread(queue.control_pending, value)
                return web.json_response(result)
        except (ValueError, TypeError, UnicodeDecodeError):
            code, status = 'queue_control_invalid_request', 400
        except QueueControlError as error:
            code, status, blocked = error.code, 409, error.blocked
        except MaintenanceError as error:
            code, status = error.code, error.status
        except (sqlite3.Error, OSError):
            logging.exception('Native queue control unavailable: %s', request_id)
        return web.json_response({'type': 'about:blank', 'title': 'Native queue control was not confirmed',
                                  'status': status, 'detail': 'Inspect the original identities and retry the same operation after transport loss.',
                                  'code': code, 'requestId': request_id, 'blocked': blocked},
                                 status=status, content_type='application/problem+json', headers={'X-Request-ID': request_id})
