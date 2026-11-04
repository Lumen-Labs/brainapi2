"""Exercise atomic pipeline storage against an explicitly disposable PostgreSQL DB."""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
import psycopg2
from psycopg2 import sql
from src.core.saving.pipeline_store import DDL, PostgreSQLCollection, predicate


def test_pipeline_query_operators_are_parameterized():
    clause, args = predicate({'task_id': "quote'", '$or': [{'status': {'$nin': ['failed']}}, {'lease': {'$exists': False}}]})
    assert "quote'" not in clause
    assert "quote'" in args[0]
    with pytest.raises(ValueError):
        predicate({'status': {'$unsupported': 1}})


@pytest.fixture
def store():
    uri = os.getenv('PROGRESSIVE_POSTGRES_TEST_URI')
    if not uri:
        pytest.skip('Disposable PostgreSQL URI was not provided')
    schema = 'pipeline_test_' + uuid.uuid4().hex
    admin = psycopg2.connect(uri)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    class Client:
        @contextmanager
        def _brain_connection(self, brain_id):
            conn = psycopg2.connect(uri, options='-c search_path=' + schema)
            try:
                yield conn
            finally:
                conn.close()
    client = Client()
    with client._brain_connection('test') as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
    try:
        yield lambda name='ingest_sections': PostgreSQLCollection(client, 'test', name)
    finally:
        with admin.cursor() as cur:
            cur.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        admin.close()


def test_atomic_claims_manifest_replay_order_and_task_uniqueness(store):
    sections = store()
    for ordinal in [10, 2, 1]:
        sections.update_one({'_id': str(ordinal)}, {'$setOnInsert': {'task_id': 't'+str(ordinal), 'import_id': 'import', 'ordinal': ordinal, 'status': 'pending'}}, upsert=True)
    sections.update_one({'_id': '1'}, {'$setOnInsert': {'status': 'failed'}}, upsert=True)
    assert [r['ordinal'] for r in sections.find({'import_id': 'import'}).sort('ordinal', 1).limit(2)] == [1, 2]
    def claim(_):
        return sections.update_one({'_id':'1','status':'pending'}, {'$set':{'status':'started'}}).modified_count
    with ThreadPoolExecutor(max_workers=8) as workers:
        assert sum(workers.map(claim, range(16))) == 1
    assert sections.find_one({'_id':'1'})['status'] == 'started'
    assert sections.count_documents({'status': {'$nin':['started']}}) == 2
    with pytest.raises(psycopg2.errors.UniqueViolation):
        sections.insert_one({'_id':'duplicate', 'task_id':'t1'})
    before = sections.find_one_and_update({'_id':'1','status':'started'}, {'$set':{'status':'completed'}})
    assert before['status'] == 'started'
    assert sections.find_one_and_update({'_id':'1','status':'started'}, {'$set':{'status':'failed'}}) is None
    sections.update_one({'_id':'1'}, {'$unset':{'task_id':''}})
    assert 'task_id' not in sections.find_one({'_id':'1'})


def test_usage_export_ack_and_dispatch_lease(store):
    events = store('pipeline_usage_events')
    for identity in ['a','b','c']:
        events.insert_one({'_id':identity, 'operation_id':'op'})
    assert events.update_many({'operation_id':'op','_id':{'$in':['a','b']}}, {'$set':{'exported':True}}).modified_count == 2
    assert [r['_id'] for r in events.find({'exported':{'$ne':True}}).sort('_id',1)] == ['c']
    roots = store('ingest_manifests')
    roots.update_one({'_id':'op'}, {'$setOnInsert':{'total':3}}, upsert=True)
    query = {'_id':'op','$or':[{'dispatch_until':{'$lt':100}}, {'dispatch_until':{'$exists':False}}]}
    assert roots.find_one_and_update(query, {'$set':{'dispatch_until':120}})['total'] == 3
    assert roots.find_one_and_update(query, {'$set':{'dispatch_until':120}}) is None
