import os
import hashlib
import json
from io import StringIO
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo_runtime import (safe_path, owns_process, free_port, is_server_process,
                                 running, stop, terminate_servers)
from model_catalog import default_model_index
from scripts.package_demo_runtime import runtime_payload, table_fingerprints
import sqlite3
import psutil


class ServerProcess:
    def __init__(self, root, pid, created=123., port=8506, *, relative=False):
        self.pid, self.created, self.alive = pid, created, True
        self.root, self.terminated, self.killed = Path(root), 0, 0
        self.command = ['python', '-m', 'streamlit', 'run',
                        '1_🔥_Home.py' if relative else str(self.root/'1_🔥_Home.py'),
                        '--server.port', str(port)]

    def cmdline(self):
        if not self.alive:
            raise psutil.NoSuchProcess(self.pid)
        return self.command

    def cwd(self):
        return str(self.root)

    def create_time(self):
        return self.created

    def is_running(self):
        return self.alive

    def terminate(self):
        self.terminated += 1
        self.alive = False

    def kill(self):
        self.killed += 1
        self.alive = False


class DemoRuntimeTests(unittest.TestCase):
    def test_sqlite_sidecars_and_live_pid_files_are_not_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('catalog.sqlite3', 'catalog.sqlite3-wal', 'catalog.sqlite3-shm', '.runtime/run/server.json', 'app.py'):
                p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture')
            self.assertEqual({p.name for p in runtime_payload(root)}, {'catalog.sqlite3', 'app.py'})

    def test_fingerprint_does_not_change_database_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'catalog.sqlite3'
            db = sqlite3.connect(path)
            db.execute('PRAGMA journal_mode=WAL'); db.execute('CREATE TABLE records(value TEXT)')
            db.execute("INSERT INTO records VALUES('human record')"); db.commit(); db.close()
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(table_fingerprints(path)['records']['rows'], 1)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_package_paths_cannot_escape_selected_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(safe_path(root, 'data/a.txt'), root.resolve()/'data/a.txt')
            for name in ('../private.txt', '.', str(root.resolve()/'absolute.txt')):
                with self.assertRaises(ValueError):
                    safe_path(root, name)

    def test_demo_model_is_explicit_and_general_model_is_unaffected(self):
        with patch.dict(os.environ, {'FIRE_DEMO_MODEL_KEY': 'trained:baseline:best'}):
            self.assertEqual(default_model_index({'fire_n': None, 'trained:baseline:best': None}), 1)
            self.assertEqual(default_model_index({'yolov8n': None}, 'yolov8n'), 0)
            with self.assertRaises(ValueError):
                default_model_index({'fire_n': None})

    def test_normal_default_remains_upstream(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_model_index({'trained:a:best': None, 'fire_n': None}), 1)

    def test_saved_pid_needs_entrypoint_and_creation_time(self):
        class Process:
            def cmdline(self):
                return ['python', '-m', 'streamlit', 'run', str(root/'1_🔥_Home.py')]
            def create_time(self):
                return 123.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(owns_process(root, {'created_at': 123.0}, Process()))
            self.assertFalse(owns_process(root, {'created_at': 124.0}, Process()))
            self.assertFalse(owns_process(root/'other', {'created_at': 123.0}, Process()))

    def test_occupied_port_is_not_reused(self):
        with socket.socket() as server:
            server.bind(('127.0.0.1', 0)); server.listen()
            port = server.getsockname()[1]
            if port > 65515:
                self.skipTest('Ephemeral port too close to upper limit')
            self.assertNotEqual(free_port(port), port)

    def test_relative_entry_uses_server_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ServerProcess(root/'another-copy', 10, relative=True)
            self.assertFalse(is_server_process(root, process))
            self.assertTrue(is_server_process(root/'another-copy', process))

    def test_entrypoint_argument_does_not_make_non_server_process_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ServerProcess(root, 10)
            process.command = ['python', 'training_worker.py', 'streamlit', 'run', str(root/'1_🔥_Home.py')]
            self.assertFalse(is_server_process(root, process))

    def test_missing_pid_record_recovers_orphan_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = ServerProcess(root, 11, created=123.02)
            other = ServerProcess(root/'other', 12)
            with patch('psutil.process_iter', return_value=[other, child]):
                state, process = running(root)
            self.assertIs(process, child)
            self.assertEqual(state['pid'], child.pid)
            self.assertEqual(state['port'], 8506)
            self.assertEqual(state['url'], 'http://127.0.0.1:8506')

    def test_stale_saved_pid_does_not_prevent_orphan_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root/'.runtime/run/server.json'
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({'pid': 10, 'created_at': 123., 'port': 8506}), encoding='utf-8')
            child = ServerProcess(root, 11, created=123.02)
            with patch('psutil.Process', side_effect=psutil.NoSuchProcess(10)), \
                 patch('psutil.process_iter', return_value=[child]):
                state, process = running(root)
            self.assertIs(process, child)
            self.assertEqual(state['pid'], 11)

    def test_stop_ends_venv_launcher_and_server_but_not_other_projects_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root/'.runtime/run/server.json'
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({'pid': 10, 'created_at': 123., 'port': 8506}), encoding='utf-8')
            launcher = ServerProcess(root, 10)
            server = ServerProcess(root, 11, created=123.02)
            other = ServerProcess(root/'other', 12)
            training = ServerProcess(root, 13)
            training.command = ['python', str(root/'scripts/training_worker.py'), 'train_123']
            processes = [launcher, server, other, training]
            with patch('psutil.Process', return_value=launcher), \
                 patch('psutil.process_iter', side_effect=lambda: processes), \
                 patch('psutil.wait_procs', side_effect=lambda procs, timeout: (list(procs), [])), \
                 patch('scripts.demo_runtime.port_open', return_value=False), \
                 patch('sys.stdout', new_callable=StringIO) as output:
                stop(root)
            self.assertEqual((launcher.terminated, server.terminated), (1, 1))
            self.assertEqual((other.terminated, training.terminated), (0, 0))
            self.assertFalse(record.exists())
            self.assertIn('端口已释放：8506', output.getvalue())

    def test_stop_does_not_claim_port_release_when_another_listener_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = ServerProcess(root, 11)
            with patch('psutil.process_iter', side_effect=lambda: [server]), \
                 patch('psutil.wait_procs', side_effect=lambda procs, timeout: (list(procs), [])), \
                 patch('scripts.demo_runtime.port_open', return_value=True), \
                 patch('sys.stdout', new_callable=StringIO) as output:
                stop(root)
            self.assertIn('仍有其他监听者', output.getvalue())
            self.assertNotIn('端口已释放', output.getvalue())

    def test_pid_reuse_before_termination_does_not_signal_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = ServerProcess(root, 11)
            with patch.object(server, 'create_time', side_effect=[123., 456.]), \
                 patch('psutil.wait_procs', return_value=([], [])):
                terminate_servers(root, [server])
            self.assertEqual((server.terminated, server.killed), (0, 0))

    def test_failed_stop_keeps_saved_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root/'.runtime/run/server.json'
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({'pid': 11, 'created_at': 123., 'port': 8506}), encoding='utf-8')
            server = ServerProcess(root, 11)
            with patch('psutil.Process', return_value=server), \
                 patch('psutil.process_iter', side_effect=lambda: [server]), \
                 patch.object(server, 'terminate'), patch.object(server, 'kill'), \
                 patch('psutil.wait_procs', side_effect=lambda procs, timeout: ([], list(procs))):
                with self.assertRaisesRegex(RuntimeError, '保留启动记录'):
                    stop(root)
            self.assertTrue(record.exists())


if __name__ == '__main__':
    unittest.main()
