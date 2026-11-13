"""Phase-only fallback for registered custom nodes with opaque inference calls."""
import inspect

from comfy.console_progress import report_progress


def instrument_custom_node(node_class, label, *, function_name=None):
    name = function_name or getattr(node_class, 'FUNCTION', None)
    if name is None:
        return
    function = inspect.getattr_static(node_class, name, None)
    descriptor = None
    if isinstance(function, (staticmethod, classmethod)):
        descriptor = type(function)
        function = function.__func__
    if not callable(function):
        return
    if getattr(function, '_console_progress', False):
        return
    wrapped = report_progress(label, delay=1)(function)
    wrapped._console_progress = True
    setattr(node_class, name, descriptor(wrapped) if descriptor else wrapped)
