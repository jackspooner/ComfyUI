"""Opt-in machine-local operator HTTP boundary; no browser or proxy access."""
import hmac
import ipaddress
import json
import os
import re
import stat
from urllib.parse import urlsplit

from aiohttp import web

from app.maintenance_store import MAX_RECEIPTS, MaintenanceError, MaintenanceStore, PromptReplay


IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def configured_store():
    directory = os.environ.get("COMFYUI_MAINTENANCE_DIR")
    token_file = os.environ.get("COMFYUI_MAINTENANCE_TOKEN_FILE")
    capacity = os.environ.get("COMFYUI_MAINTENANCE_MAX_RECEIPTS")
    if not directory and not token_file and capacity is None:
        return None, None
    if not directory or not token_file:
        raise ValueError("maintenance requires both directory and token file")
    if not os.path.isabs(token_file):
        raise ValueError("maintenance token file must be absolute")
    with os.fdopen(os.open(token_file, os.O_RDONLY | os.O_NOFOLLOW), "r") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("maintenance token file must be owner-only and owned by this user")
        token = handle.read(4097).strip()
    if not 32 <= len(token) <= 4096 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("maintenance token must contain 32-4096 non-whitespace ASCII characters")
    max_receipts = MAX_RECEIPTS
    if capacity == "unlimited":
        max_receipts = None
    elif capacity is not None:
        if not capacity.isascii() or not capacity.isdecimal() or len(capacity) > 9 or int(capacity) < 1:
            raise ValueError("COMFYUI_MAINTENANCE_MAX_RECEIPTS must be 'unlimited' or a positive integer of at most nine digits")
        max_receipts = int(capacity)
    return MaintenanceStore(directory, max_receipts=max_receipts), token


def authorize(request, token):
    if any(name in request.headers for name in ("Origin", "Sec-Fetch-Site", "Forwarded", "X-Forwarded-For", "X-Forwarded-Host")):
        raise MaintenanceError("operator_origin_forbidden", 403)
    try:
        address = ipaddress.ip_address(request.remote)
        local = address.is_loopback or (isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None and address.ipv4_mapped.is_loopback)
        host = urlsplit("//" + request.host).hostname
        host_local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        local = host_local = False
    if not local or not host_local:
        raise MaintenanceError("operator_local_only", 403)
    supplied = request.headers.get("Authorization", "")
    if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
        raise MaintenanceError("operator_unauthorized", 401)


def command_request(action, data):
    fields = {"operationId", "ownerId", "expectedRevision"}
    fields.add("targetRevision" if action == "start" else "expectedGeneration")
    if not isinstance(data, dict) or set(data) != fields:
        raise MaintenanceError("invalid_maintenance_request", 400)
    if type(data["expectedRevision"]) is not int or data["expectedRevision"] < 0:
        raise MaintenanceError("invalid_maintenance_request", 400)
    for key in fields - {"expectedRevision"}:
        if not isinstance(data[key], str) or not IDENTITY.fullmatch(data[key]):
            raise MaintenanceError("invalid_maintenance_request", 400)
    return data


def middleware(queue, token):
    @web.middleware
    async def boundary(request, handler):
        path = request.path.removeprefix("/api") if request.path.startswith("/api/") else request.path
        try:
            if path == "/maintenance" or path.startswith("/maintenance/"):
                authorize(request, token)
            if path == "/prompt" and request.method == "POST":
                with queue.admission():
                    return await handler(request)
            return await handler(request)
        except PromptReplay as replay:
            receipt = replay.receipt
            return web.json_response({"prompt_id": receipt["id"], "number": receipt["number"],
                                      "node_errors": {}, "replayed": True, "receiptState": receipt["state"]})
        except MaintenanceError as exc:
            return web.json_response({"error": {"type": exc.code, "message": exc.code,
                                                "details": "Inspect maintenance status; do not blindly resubmit.", "extra_info": {}},
                                      "accepted": exc.accepted, "node_errors": {}}, status=exc.status)
    return boundary


def add_routes(routes, queue):
    @routes.get("/maintenance")
    async def status(request):
        return web.json_response(queue.maintenance_status())

    @routes.post("/maintenance/{action:start|resume|abort|retry-publication}")
    async def command(request):
        action = request.match_info["action"]
        try:
            data = command_request(action, await request.json())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MaintenanceError("invalid_maintenance_request", 400) from exc
        result = queue.maintenance_command(action, data)
        result["command"] = {"action": action, "operationId": data["operationId"]}
        return web.json_response(result)

    @routes.get("/maintenance/receipts/{prompt_id}")
    async def receipt(request):
        with queue.mutex:
            result = queue.store.receipt(request.match_info["prompt_id"])
        if result is None:
            return web.json_response({"error": {"type": "receipt_not_found"}, "acceptance": "unknown"}, status=404)
        # Digest is internal collision evidence, not part of operator readback.
        del result["digest"]
        return web.json_response(result)
