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


start-all:
	start-milvus
	start-rabbitmq
	start-redis
	@bash -c `
		poetry run celery -A src.workers.app worker --loglevel=info
	`

stop-all:
	stop-milvus
	stop-rabbitmq
	stop-redis
	pkill -f celery

clear-all-volumes:
	delete-milvus-volumes
	delete-rabbitmq-volumes
	delete-redis-volumes