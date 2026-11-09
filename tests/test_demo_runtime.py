import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo_runtime import safe_path, owns_process, free_port
from model_catalog import default_model_index


class DemoRuntimeTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
