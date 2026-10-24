"""Evidence gates must never turn file counts or simulated roles into acceptance."""
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys
from contextlib import closing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_requirements import inspect_catalog
from check_delivery import verify_catalog_isolation


class RequirementsEvidenceTests(unittest.TestCase):
    def seed(self, path, identifiers):
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE assets(id TEXT PRIMARY KEY,kind TEXT,duration REAL)')
            db.executemany('INSERT INTO assets VALUES(?,?,?)', [(key, 'image', None) for key in identifiers])

    def test_nonempty_real_catalog_is_allowed_without_fixture_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); real = root / 'real.sqlite3'; fixture = root / 'fixture.sqlite3'
            self.seed(real, ['collected1', 'collected2']); self.seed(fixture, ['synthetic'])
            self.assertEqual(verify_catalog_isolation(real, fixture)['assets']['rows'], 2)

    def test_fixture_overlap_rejects_even_when_only_one_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); real = root / 'real.sqlite3'; fixture = root / 'fixture.sqlite3'
            self.seed(real, ['synthetic']); self.seed(fixture, ['synthetic'])
            with self.assertRaises(AssertionError):
                verify_catalog_isolation(real, fixture)

    def test_read_only_audit_does_not_initialize_missing_quality_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'catalog.sqlite3'; self.seed(path, ['real1'])
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            report = inspect_catalog(path)
            self.assertEqual(report['images'], 1)
            self.assertEqual(report['latest_approved_images'], 0)
            self.assertEqual(report['independently_double_submitted_images'], 0)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertNotIn('quality_tasks', report['tables'])

    def test_independent_pair_requires_same_task_and_distinct_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'catalog.sqlite3'; self.seed(path, ['real1'])
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('CREATE TABLE quality_submissions(task_id TEXT,asset_id TEXT,role TEXT)')
                db.executemany('INSERT INTO quality_submissions VALUES(?,?,?)', [('t1', 'real1', 'A'), ('t2', 'real1', 'B')])
            self.assertEqual(inspect_catalog(path)['independently_double_submitted_images'], 0)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("INSERT INTO quality_submissions VALUES('t1','real1','B')")
                db.execute("INSERT INTO quality_submissions VALUES('t2','real1','A')")
            self.assertEqual(inspect_catalog(path)['independently_double_submitted_images'], 1)


if __name__ == '__main__':
    unittest.main()
