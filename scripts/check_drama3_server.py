"""Isolated ComfyUI acceptance server with per-prompt resource evidence.

Uses ComfyUI's embedded server entrypoint. Set GPU visibility before launch;
pass server, model-root and output-directory CLI arguments normally. Set DRAMA3_EVIDENCE_DIRECTORY to a private
local output directory. Run only under coordinated GPU admission; this wrapper
neither acquires nor releases another process's memory or ownership.
"""
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comfy.options
comfy.options.enable_args_parsing()
import main
import execution
import torch

original_execute = execution.PromptExecutor.execute_async


async def measured_execute(self, prompt, prompt_id, *args, **kwargs):
    evidence = Path(os.environ['DRAMA3_EVIDENCE_DIRECTORY'])
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        return await original_execute(self, prompt, prompt_id, *args, **kwargs)
    finally:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result = dict(prompt_id=prompt_id, seconds=time.monotonic() - started,
                      process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(), driver_used_bytes=total - free)
        (evidence / f'{prompt_id}.resources.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    execution.PromptExecutor.execute_async = measured_execute
    loop, server, start = main.start_comfyui()
    try:
        loop.run_until_complete(start())
    except KeyboardInterrupt:
        pass
    finally:
        server.asset_manager.shutdown()
