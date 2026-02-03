"""
Shared lock so only one LLM inference runs at a time (recipe analysis or video import).
Prevents CUBLAS_STATUS_ALLOC_FAILED when GPU is used concurrently.
"""
import threading

LLM_INFERENCE_LOCK = threading.Lock()
