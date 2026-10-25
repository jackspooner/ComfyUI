"""Server queue adapter: persistence and fencing stay outside PromptExecutor."""
import contextlib
import copy
import heapq
import logging
import sqlite3

from execution import MAXIMUM_HISTORY_SIZE, PromptQueue
from app.maintenance_store import MaintenanceError


class MaintenanceQueue(PromptQueue):
    def __init__(self, server, store):
        super().__init__(server)
        self.store = store
        self.validating = 0
        self.finishing = 0
        self.publication_pending = {}
        self.history.update(store.histories())

    @contextlib.contextmanager
    def admission(self):
        with self.mutex:
            self.store.check_admission()
            self.validating += 1
        try:
            yield
        finally:
            with self.mutex:
                self.validating -= 1

    def put(self, item):
        with self.mutex:
            self.store.accept(item)
            # Durable acceptance precedes native enqueue. A crash between these
            # steps becomes unknown, never an inferred rejection or replay.
            super().put(item)

    def get(self, timeout=None):
        with self.mutex:
            # The next dequeue is the worker's acknowledgement that the prior
            # completion notifications and housekeeping have returned.
            self.finishing = 0
            while True:
                try:
                    fenced = bool(self.store.state()["active"])
                except (MaintenanceError, sqlite3.Error):
                    self.store.failure = "receipt_store_failure"
                    # A store read fault must not kill the only draining worker.
                    # Pending native work still runs; idle work stays quiescent.
                    if not self.queue:
                        self.not_empty.wait(timeout=1)
                        continue
                    fenced = True
                if not self.queue and fenced:
                    self.not_empty.wait()
                    continue
                result = super().get(timeout)
                if result is not None:
                    return result
                try:
                    fenced = bool(self.store.state()["active"])
                except (MaintenanceError, sqlite3.Error):
                    self.store.failure = "receipt_store_failure"
                    continue
                if not fenced:
                    self.finishing = 1
                    return None
                # While fenced, an empty worker stays idle. Pending accepted
                # prompts still use the ordinary consumer without any pause.

    def task_done(self, item_id, history_result, status, process_item=None):
        with self.mutex:
            self.finishing = 1
            item = self.currently_running[item_id]
            messages = []
            if status:
                for event in status.messages:
                    if isinstance(event, (tuple, list)) and len(event) == 2 and event[0] in (
                        "execution_start", "execution_success", "execution_error", "execution_interrupted"
                    ):
                        messages.append([event[0], {"prompt_id": item[1]}])
            history = {"prompt": [item[0], item[1], {}, {"create_time": item[3].get("create_time")}, item[4]],
                       "outputs": history_result.get("outputs", {}),
                       "status": {"status_str": status.status_str, "completed": status.completed, "messages": messages} if status else None}
            try:
                self.store.complete(item[1], history)
            except MaintenanceError:
                # Keep the finished result for explicit publication repair, but
                # keep consuming already accepted prompts. Never rerun a node.
                self.publication_pending[item_id] = copy.deepcopy(history)
                logging.error("Prompt completion receipt unavailable; maintenance remains blocked")
                return
            try:
                super().task_done(item_id, history_result, status, process_item)
            except Exception:
                # Native history/notification publication can fail after the
                # durable receipt committed and the running item was removed.
                self.publication_pending[item_id] = copy.deepcopy(history)
                self.store.failure = "completion_publication_failure"
                logging.error("Prompt completion publication failed; maintenance remains blocked")

    def wipe_queue(self):
        with self.mutex:
            while self.queue:
                self._removed(self.queue[0])
                heapq.heappop(self.queue)
            self.server.queue_updated()

    def _removed(self, item):
        if self.store.state()["active"]:
            raise MaintenanceError("maintenance_drain_protected")
        self.store.complete(item[1], {"prompt": [item[0], item[1], {}, {"create_time": item[3].get("create_time")}, item[4]], "outputs": {},
                                     "status": {"status_str": "error", "completed": False, "messages": [], "reason": "removed_before_execution"}})

    def delete_queue_item(self, function):
        with self.mutex:
            for index, item in enumerate(self.queue):
                if function(item):
                    self._removed(item)
                    # Base implementation calls wipe_queue for the last item;
                    # avoid recording the same removal twice.
                    self.queue.pop(index)
                    heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
            return False

    def maintenance_status(self):
        with self.mutex:
            publishing = len(self.publication_pending)
            running = len(self.currently_running.keys() - self.publication_pending.keys())
            unpublished = sum((self.store.receipt(history["prompt"][1]) or {}).get("state") == "accepted"
                              for history in self.publication_pending.values())
            return self.store.status(len(self.queue), running,
                                     self.validating, self.finishing, publishing, unpublished)

    def wipe_history(self):
        with self.mutex:
            self.store.hide_history()
            super().wipe_history()

    def delete_history_item(self, id_to_delete):
        with self.mutex:
            self.store.hide_history(id_to_delete)
            super().delete_history_item(id_to_delete)

    def retry_publication(self, request):
        self.store.check_operator(request)
        errors = []
        for item_id, history in list(self.publication_pending.items()):
            prompt_id = history["prompt"][1]
            try:
                receipt = self.store.receipt(prompt_id)
                if not receipt or receipt["state"] != "terminal":
                    self.store.complete(prompt_id, history)
                status = self.ExecutionStatus(**history["status"]) if history["status"] else None
                if item_id in self.currently_running:
                    super().task_done(item_id, {"outputs": history["outputs"]}, status,
                                      process_item=lambda item: history["prompt"])
                else:
                    if len(self.history) > MAXIMUM_HISTORY_SIZE:
                        self.history.pop(next(iter(self.history)))
                    self.history[prompt_id] = history
                    self.server.queue_updated()
                if not self.store.receipt(prompt_id)["visible"]:
                    self.history.pop(prompt_id, None)
                del self.publication_pending[item_id]
            except MaintenanceError as exc:
                errors.append({"promptId": prompt_id, "code": exc.code})
            except sqlite3.Error:
                errors.append({"promptId": prompt_id, "code": "receipt_store_failure"})
            except Exception:
                errors.append({"promptId": prompt_id, "code": "completion_publication_failure"})
        # A successful explicit probe can clear transient storage failures, not
        # missing/unknown receipts or an uncommitted completion.
        self.store.probe_write()
        self.store.failure = errors[0]["code"] if errors else None
        return errors

    def maintenance_command(self, action, request):
        with self.mutex:
            errors = None
            if action == "start":
                self.store.start(request)
            elif action == "retry-publication":
                errors = self.retry_publication(request)
            elif action in ("resume", "abort"):
                self.store.release(action, request, self.maintenance_status()["safeToStop"])
            else:
                raise MaintenanceError("invalid_maintenance_request", 400)
            self.not_empty.notify_all()
            result = self.maintenance_status()
            if errors is not None:
                result["publicationErrors"] = errors
            return result
