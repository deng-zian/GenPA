"""Optional LIBERO simulator process isolation. No CUDA initialization here.

Usage from get_libero_env:
    env = ProcessEnv({'bddl_file_name': str(task_bddl_file),
                      'camera_heights': resolution, 'camera_widths': resolution}, seed)
The factory runs the same OffScreenRenderEnv and env.seed in a spawned child.
"""
import atexit
import os
import multiprocessing
import random
import traceback
import weakref

_ACTIVE = weakref.WeakSet()


def _rng_state():
    import numpy as np
    return random.getstate(), np.random.get_state()


def _restore_rng(state):
    import numpy as np
    random.setstate(state[0])
    np.random.set_state(state[1])


def _environment_worker(connection, env_kwargs, seed, initial_rng, render_gpu):
    env = None
    try:
        # Import before restoring RNG: module imports must not consume the
        # simulator's captured parent RNG stream.
        if render_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(render_gpu)
        from libero.libero.envs import OffScreenRenderEnv
        _restore_rng(initial_rng)
        env = OffScreenRenderEnv(**env_kwargs)
        env.seed(seed)
        connection.send((True, None))
        while True:
            method, args, kwargs = connection.recv()
            try:
                if method == 'close':
                    env.close()
                    env = None
                    connection.send((True, _rng_state()))
                    break
                if method not in {'reset', 'set_init_state', 'step', 'seed'}:
                    raise ValueError('Unsupported environment method: ' + method)
                result = getattr(env, method)(*args, **kwargs)
                connection.send((True, result))
            except BaseException:
                connection.send((False, traceback.format_exc()))
                break
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        connection.close()


class ProcessEnv:
    def __init__(self, env_kwargs, seed, timeout=180.0, render_gpu=None):
        self._closed = False
        self._timeout = timeout
        initial_rng = _rng_state()
        context = multiprocessing.get_context('spawn')
        self._connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_environment_worker,
            args=(child_connection, dict(env_kwargs), seed, initial_rng, render_gpu),
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        _ACTIVE.add(self)
        try:
            self._receive()
        except BaseException:
            self._shutdown()
            raise

    def _receive(self):
        if not self._connection.poll(self._timeout):
            alive = self._process.is_alive()
            self._shutdown()
            raise TimeoutError(f'LIBERO child timed out; was_alive={alive}')
        try:
            ok, result = self._connection.recv()
        except EOFError as exc:
            exitcode = self._process.exitcode
            self._shutdown()
            raise RuntimeError(f'LIBERO child exited unexpectedly: {exitcode}') from exc
        if not ok:
            self._shutdown()
            raise RuntimeError('LIBERO child failed:\n' + result)
        return result

    def _call(self, method, *args, **kwargs):
        if self._closed:
            raise RuntimeError('LIBERO child is already closed')
        try:
            self._connection.send((method, args, kwargs))
        except (BrokenPipeError, EOFError, OSError):
            self._shutdown()
            raise
        return self._receive()

    def reset(self, *args, **kwargs):
        return self._call('reset', *args, **kwargs)

    def set_init_state(self, *args, **kwargs):
        return self._call('set_init_state', *args, **kwargs)

    def step(self, *args, **kwargs):
        return self._call('step', *args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._call('seed', *args, **kwargs)

    def close(self):
        if self._closed:
            return
        try:
            final_rng = self._call('close')
            _restore_rng(final_rng)
        finally:
            self._shutdown()

    def _shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._connection.close()
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2.0)
        _ACTIVE.discard(self)


def _close_active():
    for env in list(_ACTIVE):
        try:
            env.close()
        except BaseException:
            env._shutdown()


atexit.register(_close_active)
