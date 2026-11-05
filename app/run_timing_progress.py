"""Consume the native progress registry without changing node execution."""
from comfy_execution.progress import ProgressHandler, add_progress_handler


class TimingProgressHandler(ProgressHandler):
    def __init__(self, queue):
        super().__init__('run_timing')
        self.queue = queue

    def start_handler(self, node_id, state, prompt_id):
        with self.queue.mutex:
            self.queue.timing.safe_stage(prompt_id, node_id, 'started')

    def set_registry(self, registry):
        self.registry = registry

    def inputs_handler(self, node_id, inputs, prompt_id):
        try:
            kind = self.registry.dynprompt.get_node(node_id)['class_type']
            with self.queue.mutex:
                self.queue.timing.safe_inputs(prompt_id, node_id, kind, inputs)
        except Exception:
            self.queue.timing.issue = 'timing_input_observation_unavailable'

    def update_handler(self, node_id, value, max_value, state, prompt_id, image=None):
        with self.queue.mutex:
            self.queue.timing.safe_stage(prompt_id, node_id, 'progress', value, max_value)

    def finish_handler(self, node_id, state, prompt_id):
        with self.queue.mutex:
            self.queue.timing.safe_stage(prompt_id, node_id, 'finished')


def attach_timing_progress(queue):
    add_progress_handler(TimingProgressHandler(queue))
