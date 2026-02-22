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

build-neo4j-extension:
	docker run --rm -v $(PWD)/src/lib/neo4j:/app -w /app maven:3.9-eclipse-temurin-17 mvn compile

package-neo4j-extension:
	docker run --rm -v $(PWD)/src/lib/neo4j:/app -w /app maven:3.9-eclipse-temurin-17 mvn package -DskipTests

start-mongo:
	docker compose -f src/lib/mongo/docker-compose.yaml up -d

stop-mongo:
	docker compose -f src/lib/mongo/docker-compose.yaml down

delete-mongo-volumes:
	docker compose -f src/lib/mongo/docker-compose.yaml down -v --remove-orphans

start-api:
	ENV=development poetry run python -m uvicorn src.services.api.app:app --host 0.0.0.0 --port 8000 --access-log --log-level info --reload

stop-api:
	pkill -f "uvicorn src.services.api.app"

start-mcp:
	ENV=development poetry run python -m uvicorn src.services.mcp.app:app --host 0.0.0.0 --port 8001 --access-log --log-level info --reload

MCP_BRIDGE_DIR := mcp-stdio-http-bridge

build-mcp-bridge:
	cd $(MCP_BRIDGE_DIR) && cargo build --release

build-mcp-bridge-x86:
	cd $(MCP_BRIDGE_DIR) && cargo build --release --target x86_64-apple-darwin

start-nginx:
	docker compose -f src/services/webserver/docker-compose.yaml up -d --force-recreate

stop-mcp:
	pkill -f "uvicorn src.services.mcp.app"

DEBUG_ENVS := LANGCHAIN_DEBUG="true" LANGCHAIN_VERBOSE="true" DEBUG="true"

start-all:
	@if [ "$(filter debug,$(MAKECMDGOALS))" = "debug" ] || [ "$$DEBUG" = "true" ]; then \
		echo "DEBUG mode enabled"; \
		$(MAKE) start-milvus DEBUG=true & \
		$(MAKE) start-rabbitmq DEBUG=true & \
		$(MAKE) start-redis DEBUG=true & \
		$(MAKE) start-neo4j DEBUG=true & \
		$(MAKE) start-mongo DEBUG=true & \
		$(MAKE) start-mcp DEBUG=true & \
		ENV="development" $(MAKE) start-api DEBUG=true & \
		bash -c "export $(DEBUG_ENVS) ENV="development" && poetry run celery -A src.workers.app worker --loglevel=info --pool=threads"; \
	else \
		$(MAKE) start-milvus & \
		$(MAKE) start-rabbitmq & \
		$(MAKE) start-redis & \
		$(MAKE) start-neo4j & \
		$(MAKE) start-mongo & \
		$(MAKE) start-mcp & \
		ENV="development" $(MAKE) start-api & \
		ENV="development" poetry run celery -A src.workers.app worker --loglevel=info --pool=threads; \
	fi

debug:
	@:

stop-all:
	make stop-milvus
	make stop-rabbitmq
	make stop-redis
	make stop-neo4j
	make stop-mongo
	make stop-api
	make stop-mcp
	pkill -f celery

clear-all-volumes:
	make delete-milvus-volumes
	make delete-rabbitmq-volumes
	make delete-redis-volumes
	make delete-neo4j-volumes
	make delete-mongo-volumes

VERSION := $(shell grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
container-build:
	docker build --no-cache --platform linux/amd64 -t ghcr.io/lumen-labs/brainapi:v$(VERSION) ./
	docker tag ghcr.io/lumen-labs/brainapi:v$(VERSION) ghcr.io/lumen-labs/brainapi:latest

container-push:
	docker push ghcr.io/lumen-labs/brainapi:v$(VERSION)
	docker push ghcr.io/lumen-labs/brainapi:latest

container-release:
	BUILD_DATE=$$(date -u +'%Y-%m-%dT%H:%M:%SZ')
	BUILD_SHA=$$(git rev-parse HEAD || echo "unknown")
	CACHE_BUST=$$(date +%s)
	docker build --no-cache \
		--build-arg BUILD_DATE="$$BUILD_DATE" \
		--build-arg BUILD_SHA="$$BUILD_SHA" \
		--build-arg CACHE_BUST="$$CACHE_BUST" \
		--platform linux/amd64 \
		--label org.opencontainers.image.source="https://github.com/lumen-labs/brainapi2" \
		-t ghcr.io/lumen-labs/brainapi:v$(VERSION) ./
	docker tag ghcr.io/lumen-labs/brainapi:v$(VERSION) ghcr.io/lumen-labs/brainapi:latest
	git tag -s v$(VERSION) -m "Release v$(VERSION)"
	docker push ghcr.io/lumen-labs/brainapi:v$(VERSION)
	docker push ghcr.io/lumen-labs/brainapi:latest
	git push origin v$(VERSION)


v-patch:
	poetry version patch

v-minor:
	poetry version minor

v-major:
	poetry version major