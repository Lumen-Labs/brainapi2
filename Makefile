start-milvus:
	docker compose -f src/lib/milvus/docker-compose.yml up -d

stop-milvus:
	docker compose -f src/lib/milvus/docker-compose.yml down

delete-milvus-volumes:
	docker compose -f src/lib/milvus/docker-compose.yml down -v --remove-orphans

start-rabbitmq:
	docker compose -f src/lib/rabbitmq/docker-compose.yaml up -d

stop-rabbitmq:
	docker compose -f src/lib/rabbitmq/docker-compose.yaml down

delete-rabbitmq-volumes:
	docker compose -f src/lib/rabbitmq/docker-compose.yaml down -v --remove-orphans

start-all:
	start-milvus
	start-rabbitmq
	@bash -c `
		poetry run celery -A src.workers.app worker --loglevel=info
	`

stop-all:
	stop-milvus
	stop-rabbitmq
	pkill -f celery

clear-all-volumes:
	delete-milvus-volumes
	delete-rabbitmq-volumes