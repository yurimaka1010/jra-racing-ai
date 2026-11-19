"""Local synchronization tests; no socket, database, credentials, or model."""
from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Thread
from unittest.mock import patch
import os
import unittest
from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.db_connection import ConnectionConfig, read_only_session
from jra_racing_ai.reader_gate import reader_gate


class GateTests(unittest.TestCase):
    def test_bad_wait_budget_is_rejected(self):
        for v in (0, True, 60001):
            with self.assertRaises(SafeAuditError):
                with reader_gate(ConnectionConfig(), v):pass

    @unittest.skipUnless(os.name=='nt','Windows named mutex')
    def test_second_reader_waits_until_first_has_closed(self):
        config=replace(ConnectionConfig(),database='synthetic_gate_test')
        entered=Event();done=Event()
        def second():
            entered.set()
            with reader_gate(config,2000):done.set()
        with reader_gate(config,2000):
            worker=Thread(target=second);worker.start();self.assertTrue(entered.wait(1));self.assertFalse(done.wait(.05))
        worker.join(2);self.assertTrue(done.is_set())

    @unittest.skipUnless(os.name=='nt','Windows named mutex')
    def test_viewer_timeout_does_not_enter_a_second_connection(self):
        config=replace(ConnectionConfig(),database='synthetic_gate_timeout');observed=[]
        def second():
            try:
                with reader_gate(config,30):observed.append('unexpected')
            except SafeAuditError as e:observed.append(e.code)
        with reader_gate(config,1000):
            worker=Thread(target=second);worker.start();worker.join(1)
        self.assertEqual(observed,['query_timeout'])

    @unittest.skipUnless(os.name=='nt','Windows named mutex')
    def test_exception_releases_gate(self):
        config=replace(ConnectionConfig(),database='synthetic_gate_exception')
        with self.assertRaises(ValueError):
            with reader_gate(config,100):raise ValueError('synthetic')
        with reader_gate(config,100):pass

    def test_original_read_only_implementation_is_inside_gate(self):
        events=[]
        @contextmanager
        def gate(*args):
            events.append('gate_enter');yield;events.append('gate_exit')
        @contextmanager
        def session(*args,**kwargs):
            events.append('connect');yield 'synthetic_runner';events.append('rollback_close')
        with patch('jra_racing_ai.reader_gate.reader_gate',gate),patch('jra_racing_ai.db_connection._read_only_session',session):
            with read_only_session(ConnectionConfig(),'synthetic',gate_timeout_ms=1500) as db:
                self.assertEqual(db,'synthetic_runner');events.append('SELECT')
        self.assertEqual(events,['gate_enter','connect','SELECT','rollback_close','gate_exit'])
