import concurrent.futures

SEARCH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4
)