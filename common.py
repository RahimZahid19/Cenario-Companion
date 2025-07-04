from fastapi.responses import JSONResponse
import asyncio

def create_json_response(content, status_code=200):
    return JSONResponse(
        content=content,
        status_code=status_code,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))

def format_validation_error(exc):
    error_details = []
    for error in exc.errors():
        field_name = " -> ".join(str(loc) for loc in error["loc"])
        msg = error.get("msg", "Invalid input")
        error_details.append(f"{field_name}: {msg}")
    return error_details
