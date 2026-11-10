"""Instrument the native queue adapter without changing PromptExecutor."""
import logging
from pathlib import Path
from app.queue_control import QueueControlMixin
from app.run_timing import RunTimingStore, physical_gpu_uuid
from app.maintenance_queue import MaintenanceQueue
from execution import PromptQueue


class TimingQueueMixin:
    def put(self, item):
        with self.mutex:
            result = super().put(item)
            if self.timing:
                self.timing.safe_event('accepted', item)
            return result

    def get(self, timeout=None):
        with self.mutex:
            result = super().get(timeout)
            if result and self.timing:
                self.timing.safe_event('started', result[0])
            return result

    def task_done(self, item_id, history_result, status, process_item=None):
        with self.mutex:
            item = self.currently_running[item_id]
            if self.timing:
                self.timing.safe_event('finished', item, status)
            return super().task_done(item_id, history_result, status, process_item)

    def wipe_queue(self):
        with self.mutex:
            removed = list(self.queue)
            result = super().wipe_queue()
            if self.timing:
                for item in removed:
                    self.timing.safe_event('cancelled', item)
            return result

    def delete_queue_item(self, function):
        with self.mutex:
            before = {item[1]: item for item in self.queue}
            result = super().delete_queue_item(function)
            if self.timing:
                for prompt_id in before.keys() - {item[1] for item in self.queue}:
                    self.timing.safe_event('cancelled', before[prompt_id])
            return result


class TimedPromptQueue(QueueControlMixin, TimingQueueMixin, PromptQueue):
    pass


class TimedMaintenanceQueue(QueueControlMixin, TimingQueueMixin, MaintenanceQueue):
    def maintenance_pending(self):
        return len(self.queue) + len(self.queue_control_store.held())

    def restore_held_receipt(self, item):
        self.store.restore_held_receipt(item)

    def check_control_resume(self):
        self.store.check_admission()

    def maintenance_status(self):
        result = super().maintenance_status()
        # Held executable payloads remain retained owner work. Do not certify a
        # maintenance window that silently abandons them.
        if self.queue_control_store.held():
            result['safeToStop'] = False
            result['blockers'].append('queued_work_paused')
            if result['operationId']:
                result['phase'] = 'blocked'
        return result


def create_timed_queue(server, maintenance_store, filename):
    queue = TimedMaintenanceQueue(server, maintenance_store) if maintenance_store else TimedPromptQueue(server)
    queue.configure_queue_control(str(Path(filename).with_name('queue-controls.sqlite')))
    try:
        queue.timing = RunTimingStore(filename, gpu_uuid=physical_gpu_uuid())
        from app.run_timing_metrics import attach_runtime_metrics
        attach_runtime_metrics(queue)
    except Exception:
        logging.warning('Run timing store unavailable at startup; queue remains usable', exc_info=False)
        queue.timing = None
    return queue
