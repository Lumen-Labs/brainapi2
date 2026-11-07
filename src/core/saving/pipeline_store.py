"""Durable pipeline documents for MongoDB and the PostgreSQL light profile.

This intentionally implements only the collection operations used by the import
coordinator. PostgreSQL mutations evaluate their predicates inside the database,
so claims, compare-and-swap updates, and leases are atomic across processes.
"""
import json
from types import SimpleNamespace

import psycopg2.extras

DDL = """
CREATE TABLE IF NOT EXISTS pipeline_documents (
    namespace TEXT NOT NULL,
    id TEXT NOT NULL,
    document JSONB NOT NULL,
    PRIMARY KEY (namespace, id)
);
CREATE INDEX IF NOT EXISTS pipeline_documents_import
    ON pipeline_documents (namespace, (document->>'import_id'));
CREATE INDEX IF NOT EXISTS pipeline_documents_operation
    ON pipeline_documents (namespace, (document->>'operation_id'));
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_documents_task
    ON pipeline_documents ((document->>'task_id'))
    WHERE namespace = 'ingest_sections';
"""
NAMES = frozenset({'ingest_manifests', 'ingest_sections', 'ingest_runs', 'ingest_access_gates', 'ingest_staged_inputs', 'ingest_staged_outputs', 'pipeline_usage_events'})


def supports_storage(client):
    return callable(getattr(client, 'get_collection', None)) or callable(getattr(client, '_brain_connection', None))


def collection(brain_id, name):
    from src.services.data.main import data_adapter
    client = data_adapter.data
    if name not in NAMES:
        raise ValueError('Unsupported pipeline collection')
    if callable(getattr(client, 'get_collection', None)):
        return client.get_collection(name, database=brain_id)
    return PostgreSQLCollection(client, brain_id, name)


def predicate(query):
    parts, values = [], []
    for key, value in query.items():
        if key == '$or':
            clauses = [predicate(item) for item in value]
            parts.append('(' + ' OR '.join(sql for sql, _ in clauses) + ')')
            for _, args in clauses:
                values.extend(args)
            continue
        if isinstance(value, dict):
            for op, operand in value.items():
                if op == '$exists':
                    parts.append('(document ? %s) = %s')
                    values.extend([key, bool(operand)])
                elif op == '$ne':
                    parts.append('(document->%s) IS DISTINCT FROM %s::jsonb')
                    values.extend([key, json.dumps(operand)])
                elif op in ('$in', '$nin'):
                    choices = operand
                    inner = ' OR '.join('(document->%s) = %s::jsonb' for _ in choices) or 'FALSE'
                    parts.append(('NOT COALESCE((' + inner + '), FALSE)') if op == '$nin' else '(' + inner + ')')
                    for item in choices:
                        values.extend([key, json.dumps(item)])
                elif op == '$lt':
                    parts.append('(document->>%s)::numeric < %s')
                    values.extend([key, operand])
                else:
                    raise ValueError('Unsupported pipeline predicate')
        else:
            parts.append('document @> %s::jsonb')
            values.append(json.dumps({key: value}))
    return ' AND '.join(parts) or 'TRUE', values


def update_expression(update):
    if any(key not in {'$set', '$unset', '$setOnInsert'} for key in update):
        raise ValueError('Unsupported pipeline mutation')
    return '(document || %s::jsonb) - %s::text[]', [json.dumps(update.get('$set', {})), list(update.get('$unset', {}))]


class Cursor:
    def __init__(self, collection, query):
        self.collection, self.query = collection, query
        self.order, self.maximum = None, None

    def sort(self, key, direction):
        self.order = (key, direction)
        return self

    def limit(self, count):
        self.maximum = max(0, int(count))
        return self

    def __iter__(self):
        sql, args = predicate(self.query)
        sql = 'SELECT document FROM pipeline_documents WHERE namespace = %s AND (' + sql + ')'
        args = [self.collection.name, *args]
        if self.order:
            # Ordinals must sort numerically, not lexicographically.
            field, direction = self.order
            expression = '(document->>%s)::bigint' if field == 'ordinal' else 'document->>%s'
            sql += ' ORDER BY ' + expression + (' DESC' if direction < 0 else ' ASC')
            args.append(field)
        if self.maximum is not None:
            sql += ' LIMIT %s'
            args.append(self.maximum)
        with self.collection.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                rows = [row[0] for row in cur.fetchall()]
        return iter(rows)


class PostgreSQLCollection:
    def __init__(self, client, brain_id, name):
        self.client, self.brain_id, self.name = client, brain_id, name

    def connection(self):
        return self.client._brain_connection(self.brain_id)

    def create_index(self, *args, **kwargs):
        # The schema owns the import/operation and unique section-task indexes.
        return None

    def find(self, query):
        return Cursor(self, query)

    def find_one(self, query):
        return next(iter(self.find(query).limit(1)), None)

    def count_documents(self, query):
        sql, args = predicate(query)
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM pipeline_documents WHERE namespace=%s AND (' + sql + ')', [self.name, *args])
                return cur.fetchone()[0]

    def delete_many(self, query):
        where, args = predicate(query)
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'DELETE FROM pipeline_documents WHERE namespace = %s AND (' + where + ')',
                    [self.name, *args],
                )
                deleted = cur.rowcount
            conn.commit()
        return SimpleNamespace(deleted_count=deleted)

    def insert_one(self, document):
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute('INSERT INTO pipeline_documents (namespace,id,document) VALUES (%s,%s,%s::jsonb)', [self.name, document['_id'], json.dumps(document)])
            conn.commit()
        return SimpleNamespace(inserted_id=document['_id'])

    def _update(self, query, update, many=False, previous=False, upsert=False):
        where, where_args = predicate(query)
        expr, expr_args = update_expression(update)
        with self.connection() as conn:
            with conn.cursor() as cur:
                if upsert:
                    if set(query) != {'_id'}:
                        raise ValueError('Pipeline upserts require an explicit identity')
                    document = {**query, **update.get('$setOnInsert', {}), **update.get('$set', {})}
                    cur.execute('INSERT INTO pipeline_documents (namespace,id,document) VALUES (%s,%s,%s::jsonb) ON CONFLICT DO NOTHING RETURNING id', [self.name, query['_id'], json.dumps(document)])
                    if cur.fetchone():
                        conn.commit()
                        return SimpleNamespace(modified_count=0, matched_count=0, upserted_id=query['_id'])
                cur.execute('SELECT id, document FROM pipeline_documents WHERE namespace=%s AND (' + where + ')' + ('' if many else ' LIMIT 1') + ' FOR UPDATE', [self.name, *where_args])
                rows = cur.fetchall()
                changed = 0
                for identity, document in rows:
                    cur.execute('UPDATE pipeline_documents SET document=' + expr + ' WHERE namespace=%s AND id=%s AND document IS DISTINCT FROM ' + expr, [*expr_args, self.name, identity, *expr_args])
                    changed += cur.rowcount
            conn.commit()
        if previous:
            return rows[0][1] if rows else None
        return SimpleNamespace(modified_count=changed, matched_count=len(rows))

    def update_one(self, query, update, upsert=False):
        return self._update(query, update, upsert=upsert)

    def update_many(self, query, update):
        return self._update(query, update, many=True)

    def find_one_and_update(self, query, update):
        return self._update(query, update, previous=True)
