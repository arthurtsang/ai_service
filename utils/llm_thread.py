"""
Run blocking LLM work in a thread so the event loop can serve other requests
(status checks, health, etc.). Use this for any sync code that calls model.generate().

Why not more Uvicorn workers: each worker loads its own copy of the model (2x+ RAM),
and import job state (_import_jobs) is per-process, so status polls would need
shared storage (e.g. Redis) to work across workers.
"""
import asyncio


async def run_llm_in_thread(func, *args, **kwargs):
    """Run blocking LLM work in a thread. Use for any sync function that calls model.generate()."""
    return await asyncio.to_thread(func, *args, **kwargs)
