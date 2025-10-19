start-milvus:
	docker compose -f src/lib/milvus/docker-compose.yml up -d

stop-milvus:
	docker compose -f src/lib/milvus/docker-compose.yml down

delete-milvus-volumes:
	docker compose -f src/lib/milvus/docker-compose.yml down -v --remove-orphans