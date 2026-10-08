# Owner(s): ["oncall: distributed"]

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import patch

from torch.distributed._transport import wait_all
from torch.distributed._transport.nixl._work import _FutureProgress, _PollingWork
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _ManualWork(_PollingWork):
    def __init__(self, *, done=False, error=None, progress=None):
        super().__init__(_FutureProgress() if progress is None else progress)
        self.done = done
        self._error = error
        self.polls = 0

    def _poll(self):
        self.polls += 1
        return self.done


@instantiate_parametrized_tests
class TestNIXLPollingWork(TestCase):
    def test_polling_backoff(self):
        work = _ManualWork()
        with patch(
            "torch.distributed._transport.nixl._work.time.sleep",
            side_effect=lambda _: setattr(work, "done", True),
        ) as sleep:
            work.wait()
            sleep.assert_called_once_with(0.001)

    def test_wait_timeout_and_retry(self):
        work = _ManualWork()
        with self.assertRaises(TimeoutError):
            work.wait(timedelta(milliseconds=1))
        self.assertFalse(work.is_completed())
        work.done = True
        self.assertTrue(work.wait())
        self.assertTrue(work.is_success())
        self.assertEqual(work.result(), [])
        self.assertEqual(work.get_future().wait(), [])

    def test_failure_is_terminal(self):
        work = _ManualWork(done=True, error=RuntimeError("native failure"))
        self.assertTrue(work.is_completed())
        self.assertFalse(work.is_success())
        self.assertIsInstance(work.exception(), RuntimeError)
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            work.wait()
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            work.get_future().wait()

    @parametrize("failed", [False, True])
    def test_future_without_event_loop(self, failed):
        error = RuntimeError("native failure") if failed else None
        work = _ManualWork(error=error)
        ready = threading.Event()
        future = work.get_future()
        worker = work._progress._thread
        future.add_done_callback(lambda _: ready.set())
        self.assertIs(work.get_future(), future)
        self.assertFalse(future.done())
        work.done = True
        self.assertTrue(ready.wait(5))
        if failed:
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                future.wait()
        else:
            self.assertEqual(future.then(lambda f: f.wait()).wait(), [])
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(work._progress._thread)

    @parametrize("failure", ["construction", "start"])
    def test_future_thread_failure_can_be_retried(self, failure):
        work = _ManualWork()
        target = "Lock" if failure == "construction" else "Thread.start"
        with patch(
            f"torch.distributed._transport.nixl._work.threading.{target}",
            side_effect=RuntimeError("thread failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread failed"):
                work.get_future()
        self.assertIsNone(work._progress._thread)
        self.assertEqual(work._progress._works, {})
        ready = threading.Event()
        future = work.get_future()
        worker = work._progress._thread
        self.assertIsNotNone(worker)
        future.add_done_callback(lambda _: ready.set())
        self.assertIs(work.get_future(), future)
        try:
            work.done = True
            self.assertTrue(ready.wait(5))
            self.assertEqual(future.wait(), [])
        finally:
            work.done = True
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(work._progress._thread)

    def test_future_outlives_event_loop(self):
        work = _ManualWork()

        async def request():
            return work.get_future()

        future = asyncio.run(request())
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        work.done = True
        self.assertTrue(ready.wait(5))
        self.assertEqual(future.wait(), [])

    def test_concurrent_future_requests_share_worker(self):
        progress = _FutureProgress()
        works = [_ManualWork(progress=progress) for _ in range(2)]
        barrier = threading.Barrier(4)

        def request(index):
            barrier.wait(timeout=5)
            return works[index % 2].get_future(), progress._thread

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(request, range(4)))
            worker = results[0][1]
            for index, (future, thread) in enumerate(results):
                self.assertIs(future, works[index % 2].get_future())
                self.assertIs(thread, worker)
        finally:
            for work in works:
                work.done = True
        worker.join(5)
        self.assertFalse(worker.is_alive())
        for work in works:
            self.assertTrue(work.get_future().done())

        work = _ManualWork(progress=progress)
        future = work.get_future()
        next_worker = progress._thread
        self.assertIsNot(next_worker, worker)
        work.done = True
        next_worker.join(5)
        self.assertFalse(next_worker.is_alive())
        self.assertTrue(future.done())

    def test_callback_registers_pending_future(self):
        progress = _FutureProgress()
        first = _ManualWork(progress=progress)
        second = _ManualWork(progress=progress)
        requested = threading.Event()
        ready = threading.Event()

        def request(_):
            second.get_future().add_done_callback(lambda _: ready.set())
            requested.set()

        first.get_future().add_done_callback(request)
        first.done = True
        try:
            self.assertTrue(requested.wait(5))
        finally:
            second.done = True
        self.assertTrue(ready.wait(5))

    @parametrize("failed", [False, True])
    def test_future_notifies_event_loop(self, failed):
        error = RuntimeError("native failure") if failed else None
        work = _ManualWork(error=error)

        async def run():
            loop = asyncio.get_running_loop()
            caller = threading.get_ident()
            future = work.get_future()
            callback = loop.create_future()

            def notify(_):
                callback.set_result((threading.get_ident(), asyncio.get_running_loop()))

            future.add_done_callback(notify)
            with ThreadPoolExecutor(max_workers=1) as executor:
                requested = executor.submit(work.get_future).result(timeout=5)
                self.assertIs(requested, future)
            self.assertIs(work.get_future(), future)
            self.assertFalse(future.done())
            work.done = True
            thread, callback_loop = await asyncio.wait_for(callback, 5)
            self.assertEqual(thread, caller)
            self.assertIs(callback_loop, loop)
            if failed:
                with self.assertRaisesRegex(RuntimeError, "native failure"):
                    future.wait()
            else:
                self.assertEqual(future.wait(), [])

        try:
            asyncio.run(run())
        finally:
            work.done = True
            worker = work._progress._thread
            if worker is not None:
                worker.join(5)
                self.assertFalse(worker.is_alive())

    @parametrize("close_loop", [False, True])
    def test_future_outlives_loop_with_queued_poll(self, close_loop):
        work = _ManualWork()
        loop = asyncio.new_event_loop()
        queued = threading.Event()
        ready = threading.Event()
        call_soon = loop.call_soon_threadsafe
        futures = []

        def schedule(callback, *args, **kwargs):
            handle = call_soon(callback, *args, **kwargs)
            queued.set()
            return handle

        def request():
            future = work.get_future()
            future.add_done_callback(lambda _: ready.set())
            futures.append(future)
            queued.wait(5)
            loop.stop()

        try:
            with patch.object(loop, "call_soon_threadsafe", side_effect=schedule):
                loop.call_soon(request)
                loop.run_forever()
            self.assertTrue(queued.is_set())
            if close_loop:
                loop.close()
            work.done = True
            self.assertTrue(ready.wait(5))
            self.assertEqual(futures[0].wait(), [])
            if not close_loop:
                loop.call_soon(loop.stop)
                loop.run_forever()
        finally:
            work.done = True
            loop.close()
            worker = work._progress._thread
            if worker is not None:
                worker.join(5)
                self.assertFalse(worker.is_alive())

    def test_future_reentrant_callback(self):
        async def run():
            work = _ManualWork()
            future = work.get_future()
            seen = []
            future.add_done_callback(lambda _: seen.append(work.wait()))
            work.done = True
            await wait_all([work])
            self.assertEqual(seen, [True])

        asyncio.run(run())

    def test_async_wait_yields(self):
        async def run():
            work = _ManualWork()

            async def complete():
                await asyncio.sleep(0.01)
                work.done = True

            task = asyncio.create_task(complete())
            await wait_all([work], timeout=1)
            await task
            self.assertGreater(work.polls, 1)

        asyncio.run(run())

    def test_timeout_does_not_complete_work(self):
        async def run():
            work = _ManualWork()
            with self.assertRaises(TimeoutError):
                await wait_all([work], timeout=0.001)
            self.assertFalse(work.is_completed())
            work.done = True
            await wait_all([work])

        asyncio.run(run())

    def test_cancellation_does_not_complete_work(self):
        async def run():
            work = _ManualWork()
            task = asyncio.create_task(wait_all([work]))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(work.is_completed())
            work.done = True
            await wait_all([work])

        asyncio.run(run())

    def test_error_drains_other_work(self):
        async def run():
            good = _ManualWork()
            bad = _ManualWork(done=True, error=ValueError("bad"))
            asyncio.get_running_loop().call_later(0.01, setattr, good, "done", True)
            with self.assertRaisesRegex(ValueError, "bad"):
                await wait_all([bad, good], timeout=1)
            self.assertTrue(good.is_completed())

        asyncio.run(run())

    def test_generator_error_drains_submitted_work(self):
        work = _ManualWork()

        def items():
            yield work
            raise ValueError("generator failed")

        async def run():
            asyncio.get_running_loop().call_later(0.01, setattr, work, "done", True)
            with self.assertRaisesRegex(ValueError, "generator failed"):
                await wait_all(items(), timeout=1)
            self.assertTrue(work.is_completed())

        asyncio.run(run())


if __name__ == "__main__":
    run_tests()
