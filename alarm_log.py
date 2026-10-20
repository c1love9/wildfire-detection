"""SQLite event history and transition recording; no per-frame event spam."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from alarm import AlarmConfig, AlarmEngine, CLASS_NAMES, LEVEL_NAMES


END_NAMES = {'recovered': '已恢复', 'stopped': '手动停止', 'completed': '视频结束', 'error': '检测异常'}


def alarm_log_path(root):
    return Path(os.environ.get('FIRE_ALARM_LOG_PATH', Path(root) / 'artifacts' / 'alarms' / 'alarms.sqlite3'))


def alarm_media_root(log_path):
    return Path(os.environ.get('FIRE_ALARM_MEDIA_ROOT', Path(log_path).parent / 'media')).resolve()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


class AlarmLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at TEXT NOT NULL,
                    ended_at TEXT, end_reason TEXT, config_json TEXT NOT NULL, metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    started_at TEXT NOT NULL, ended_at TEXT,
                    start_frame INTEGER NOT NULL, start_seconds REAL NOT NULL,
                    end_frame INTEGER, end_seconds REAL,
                    last_level INTEGER NOT NULL, peak_level INTEGER NOT NULL,
                    classes_json TEXT NOT NULL, max_confidence REAL NOT NULL, end_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL REFERENCES events(id),
                    recorded_at TEXT NOT NULL, frame INTEGER NOT NULL, media_seconds REAL NOT NULL,
                    from_level INTEGER NOT NULL, to_level INTEGER NOT NULL, kind TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_started ON events(started_at DESC);
                CREATE INDEX IF NOT EXISTS transitions_event ON transitions(event_id, id);
                CREATE TABLE IF NOT EXISTS event_media (
                    event_id TEXT PRIMARY KEY REFERENCES events(id), snapshot_path TEXT, clip_path TEXT,
                    status TEXT NOT NULL, details_json TEXT NOT NULL, error TEXT, updated_at TEXT NOT NULL
                );
            ''')

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def start_run(self, run_id, source, config, metadata):
        with self._connection() as connection:
            connection.execute('INSERT INTO runs(id, source, started_at, config_json, metadata_json) VALUES(?,?,?,?,?)',
                               (run_id, source, _now(), json.dumps(asdict(config)), json.dumps(metadata)))

    def save_event(self, event, transition):
        with self._connection() as connection:
            connection.execute('''INSERT INTO events VALUES(
                :id,:run_id,:started_at,:ended_at,:start_frame,:start_seconds,:end_frame,:end_seconds,
                :last_level,:peak_level,:classes_json,:max_confidence,:end_reason)
                ON CONFLICT(id) DO UPDATE SET ended_at=excluded.ended_at, end_frame=excluded.end_frame,
                    end_seconds=excluded.end_seconds, last_level=excluded.last_level, peak_level=excluded.peak_level,
                    classes_json=excluded.classes_json, max_confidence=excluded.max_confidence, end_reason=excluded.end_reason''', event)
            if transition:
                connection.execute('''INSERT INTO transitions(event_id, recorded_at, frame, media_seconds, from_level, to_level, kind)
                    VALUES(:event_id,:recorded_at,:frame,:media_seconds,:from_level,:to_level,:kind)''', transition)

    def finish_run(self, run_id, reason):
        with self._connection() as connection:
            connection.execute('UPDATE runs SET ended_at=?, end_reason=? WHERE id=?', (_now(), reason, run_id))

    def sources(self):
        with self._connection() as connection:
            return [row[0] for row in connection.execute('SELECT DISTINCT source FROM runs ORDER BY source')]

    def events(self, source=None, peak_level=None, limit=500):
        clauses, parameters = [], []
        if source is not None:
            clauses.append('r.source=?')
            parameters.append(source)
        if peak_level is not None:
            clauses.append('e.peak_level=?')
            parameters.append(peak_level)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                'SELECT e.*, r.source FROM events e JOIN runs r ON e.run_id=r.id' + where +
                ' ORDER BY e.started_at DESC, e.id DESC LIMIT ?', [*parameters, limit])]

    def transitions(self, event_id):
        with self._connection() as connection:
            return [dict(row) for row in connection.execute('SELECT * FROM transitions WHERE event_id=? ORDER BY id', (event_id,))]

    def save_media(self, event_id, snapshot_path, clip_path, status, details, error=None):
        with self._connection() as connection:
            connection.execute('''INSERT INTO event_media VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO UPDATE SET snapshot_path=excluded.snapshot_path, clip_path=excluded.clip_path,
                    status=excluded.status, details_json=excluded.details_json, error=excluded.error, updated_at=excluded.updated_at''',
                               (event_id, snapshot_path, clip_path, status, json.dumps(details), error, _now()))

    def media(self, event_id):
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM event_media WHERE event_id=?', (event_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result['details'] = json.loads(result.pop('details_json'))
        return result


def event_rows(events):
    return [{
        '事件编号': event['id'], '视频': event['source'], '记录时间（UTC）': event['started_at'],
        '起始帧': event['start_frame'], '开始视频时间（秒）': round(event['start_seconds'], 3),
        '结束视频时间（秒）': round(event['end_seconds'], 3) if event['end_seconds'] is not None else None,
        '类别': '、'.join(CLASS_NAMES[label] for label in json.loads(event['classes_json'])),
        '最高等级': LEVEL_NAMES[event['peak_level']], '最后等级': LEVEL_NAMES[event['last_level']],
        '最高置信度': round(event['max_confidence'], 3),
        '结束原因': END_NAMES.get(event['end_reason'], '未结束'),
    } for event in events]


def events_csv(events):
    rows = event_rows(events)
    output = StringIO(newline='')
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            # Uploaded filenames are user content; keep them literal in Excel.
            writer.writerow({key: "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@')) else value for key, value in row.items()})
    return output.getvalue().encode('utf-8-sig')


class AlarmMonitor:
    def __init__(self, config: AlarmConfig, source: str, log_path=None, metadata=None):
        self.engine = AlarmEngine(config)
        self.run_id = uuid.uuid4().hex
        self.source = source
        self.current_event = None
        self.recent_transitions = deque(maxlen=50)
        self.event_count = 0
        self.last_frame, self.last_seconds = 0, 0.0
        self.finished = False
        self.log_error = None
        self.store = None
        if log_path is not None:
            try:
                self.store = AlarmLog(log_path)
                self.store.start_run(self.run_id, source, config, metadata or {})
            except (sqlite3.Error, OSError) as exc:
                raise RuntimeError('无法创建报警日志，请检查日志目录权限与可用空间。') from exc

    @property
    def snapshot(self):
        return self.engine.snapshot

    def _write(self, transition):
        if self.store:
            try:
                self.store.save_event(self.current_event, transition)
            except (sqlite3.Error, OSError):
                self.log_error = '报警日志写入失败，请检查磁盘空间或权限；当前检测继续，但部分历史可能未保存。'

    def _transition(self, old, new, kind):
        transition = dict(event_id=self.current_event['id'], recorded_at=_now(), frame=self.last_frame,
                          media_seconds=self.last_seconds, from_level=old, to_level=new, kind=kind)
        self.recent_transitions.append(transition)
        self._write(transition)

    def update(self, number, seconds, detections):
        if self.finished:
            raise RuntimeError('已结束的报警会话不能继续计数。')
        state = self.engine.update(number, seconds, detections)
        self.last_frame, self.last_seconds = number, seconds
        if state.level and self.current_event is None:
            self.current_event = dict(id=uuid.uuid4().hex, run_id=self.run_id, started_at=_now(), ended_at=None,
                                      start_frame=number, start_seconds=seconds, end_frame=None, end_seconds=None,
                                      last_level=state.level, peak_level=state.level, classes_json='[]',
                                      max_confidence=0.0, end_reason=None)
            self.event_count += 1
        if self.current_event:
            event = self.current_event
            event['last_level'] = state.level
            event['peak_level'] = max(event['peak_level'], state.level)
            event['max_confidence'] = max(event['max_confidence'], state.max_confidence)
            event['classes_json'] = json.dumps(sorted(set(json.loads(event['classes_json'])) | set(state.classes)))
            if state.level != state.previous_level:
                if not state.level:
                    event.update(ended_at=_now(), end_frame=number, end_seconds=seconds, end_reason='recovered')
                kind = 'recovered' if not state.level else 'confirmed' if not state.previous_level else 'escalated' if state.level > state.previous_level else 'downgraded'
                self._transition(state.previous_level, state.level, kind)
                if not state.level:
                    self.current_event = None
        return state

    def finish(self, reason):
        if self.finished:
            return
        self.finished = True
        if self.current_event:
            self.current_event.update(ended_at=_now(), end_frame=self.last_frame, end_seconds=self.last_seconds, end_reason=reason)
            # Stop/EOF/error ends observation, not the fire. Keep its last level.
            self._transition(self.snapshot.level, self.snapshot.level, reason)
            self.current_event = None
        if self.store:
            try:
                self.store.finish_run(self.run_id, reason)
            except (sqlite3.Error, OSError):
                self.log_error = '报警会话结束记录未能写入，请检查磁盘空间或权限。'
