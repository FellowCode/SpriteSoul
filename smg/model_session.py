"""Sequential JSON-lines worker sessions for models with separate Python runtimes."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


RESPONSE_PREFIX = "SPRITE_SOUL_RESPONSE "


def serve(handler) -> None:
    """Keep one model alive while handling requests; ordinary output is progress."""
    for line in sys.stdin:
        try:
            handler(json.loads(line))
            response = {"ok": True}
        except Exception as exc:
            traceback.print_exc()
            response = {"ok": False, "error": str(exc)}
        gc.collect()
        print(RESPONSE_PREFIX + json.dumps(response, ensure_ascii=True), flush=True)


class ModelSession:
    def __init__(self, python: Path, worker: Path, root: Path):
        self.command = [str(python), "-u", "-X", "faulthandler", str(worker), str(root), "--serve"]
        self.root = root
        self.process = None

    def run(self, arguments: list[str], progress=None) -> None:
        if self.process is None:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(
                self.command, cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        process = self.process
        tail = []
        try:
            process.stdin.write(json.dumps(arguments, ensure_ascii=True) + "\n")
            process.stdin.flush()
            for line in process.stdout:
                line = line.rstrip()
                if line.startswith(RESPONSE_PREFIX):
                    response = json.loads(line[len(RESPONSE_PREFIX):])
                    if not response["ok"]:
                        raise RuntimeError(response["error"])
                    return
                if line:
                    tail = (tail + [line])[-10:]
                    if progress:
                        progress(line)
            code = process.wait() & 0xFFFFFFFF
            raise RuntimeError(f"AI worker завершился (код 0x{code:08X}): " + " | ".join(tail))
        except (BrokenPipeError, OSError):
            self.close()
            raise
        finally:
            if process.poll() is not None:
                self.close()

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
