import os
import io
import queue
import threading
import time


class AsyncModelSaver:
    """
    主线程: model.save(BytesIO)  —— CPU 序列化, 一般 < 50ms
    后台线程: BytesIO -> 磁盘    —— 慢 IO 全在这里
    """
    def __init__(self, max_queue=4, verbose=1):
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._verbose = verbose
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="AsyncModelSaver"
        )
        self._worker.start()

    def submit(self, model, path_without_ext: str):
        t0 = time.perf_counter()
        buf = io.BytesIO()
        try:
            model.save(buf)
        except Exception as e:
            print(f"[async] AsyncModelSaver 内存序列化失败: {e}")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self._verbose >= 2:
            print(f"[saver] 主线程序列化耗时 {elapsed_ms:.1f} ms, "
                  f"大小 {len(buf.getvalue())/1024:.1f} KB")

        buf.seek(0)
        task = (buf, path_without_ext)
        try:
            self._q.put_nowait(task)
        except queue.Full:
            try:
                dropped = self._q.get_nowait()
                self._q.task_done()
                if self._verbose:
                    print(f"[saver] 队列满，丢弃旧任务 -> {dropped[1]}")
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(task)
            except queue.Full:
                print("[async] AsyncModelSaver 队列异常满载，丢弃本次 save")

    def _run(self):
        while not self._stop.is_set():
            try:
                buf, path_no_ext = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            target_path = path_no_ext if path_no_ext.endswith(".zip") else path_no_ext + ".zip"
            tmp_path = target_path + ".tmp"
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                t0 = time.perf_counter()
                with open(tmp_path, "wb") as f:
                    f.write(buf.getvalue())
                os.replace(tmp_path, target_path)
                elapsed = time.perf_counter() - t0
                if self._verbose:
                    print(f"[async] 已落盘 {target_path} ({elapsed*1000:.0f} ms)")
            except Exception as e:
                print(f"[async] 写盘失败 {target_path}: {e}")
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            finally:
                self._q.task_done()

    def flush(self, timeout=60):
        deadline = time.time() + timeout
        while not self._q.empty():
            if time.time() > deadline:
                print("[async] AsyncModelSaver.flush 超时")
                return False
            time.sleep(0.1)
        self._q.join()
        return True

    def close(self, timeout=60):
        self.flush(timeout=timeout)
        self._stop.set()
        self._worker.join(timeout=5)
