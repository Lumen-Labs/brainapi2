start-milvus:
	docker compose -f src/lib/milvus/docker-compose.yaml up -d

stop-milvus:
	docker compose -f src/lib/milvus/docker-compose.yaml down

delete-milvus-volumes:
	docker compose -f src/lib/milvus/docker-compose.yaml down -v --remove-orphans

start-rabbitmq:
	docker compose -f src/lib/rabbitmq/docker-compose.yaml up -d

stop-rabbitmq:
	docker compose -f src/lib/rabbitmq/docker-compose.yaml down

delete-rabbitmq-volumes:
	docker compose -f src/lib/rabbitmq/docker-compose.yaml down -v --remove-orphans

start-redis:
	docker compose -f src/lib/redis/docker-compose.yaml up -d

stop-redis:
	docker compose -f src/lib/redis/docker-compose.yaml down

delete-redis-volumes:
	docker compose -f src/lib/redis/docker-compose.yaml down -v --remove-orphans


start-neo4j:
	docker compose -f src/lib/neo4j/docker-compose.yaml up -d

stop-neo4j:
	docker compose -f src/lib/neo4j/docker-compose.yaml down

delete-neo4j-volumes:
	docker compose -f src/lib/neo4j/docker-compose.yaml down -v --remove-orphans

start-api:
	poetry run uvicorn src.services.api.app:app --host 0.0.0.0 --port 8000

stop-api:
	pkill -f uvicorn


start-all:
	export ENV=development
	make start-milvus &
	make start-rabbitmq &
	make start-redis &
	make start-neo4j &
	make start-api &
	@bash -c "poetry run celery -A src.workers.app worker --loglevel=info"

stop-all:
	make stop-milvus
	make stop-rabbitmq
	make stop-redis
	make stop-neo4j
	make stop-api
	pkill -f celery

clear-all-volumes:
	make delete-milvus-volumes
	make delete-rabbitmq-volumes
	make delete-redis-volumes
	make delete-neo4j-volumes