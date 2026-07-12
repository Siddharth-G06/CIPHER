# ============================================================
# CIPHER — Fraud Detection System
# Makefile (Module 10)
# ============================================================
#
# Usage:
#   make up           Start all services in detached mode
#   make down         Stop all services
#   make clean        Stop all services and remove volumes
#   make logs         Tail cipher-app logs
#   make train        Run model trainer inside container
#   make test         Run full test suite inside container
#   make kafka-lag    Show consumer group lag
#   make kafka-topics List all Kafka topics
#   make rebuild      Rebuild cipher-app image and restart
# ============================================================

.PHONY: up down clean logs train test kafka-lag kafka-topics rebuild

## Start all services in detached mode
up:
	docker-compose up -d

## Stop all services (keep volumes)
down:
	docker-compose down

## Stop all services and remove named volumes (destructive — clears all data)
clean:
	docker-compose down -v

## Tail cipher-app logs (Ctrl+C to stop)
logs:
	docker-compose logs -f cipher-app

## Run model trainer inside the running cipher-app container
train:
	docker exec cipher-app python src/ml_layer/trainer.py

## Run full test suite inside the running cipher-app container
test:
	docker exec cipher-app pytest tests/ -v

## Show consumer group lag for cipher-fraud-detector group
kafka-lag:
	docker exec cipher-kafka kafka-consumer-groups \
		--bootstrap-server localhost:9092 \
		--describe \
		--group cipher-fraud-detector

## List all Kafka topics
kafka-topics:
	docker exec cipher-kafka kafka-topics \
		--bootstrap-server localhost:9092 \
		--list

## Rebuild cipher-app image and restart it (without touching kafka/mlflow)
rebuild:
	docker-compose build cipher-app
	docker-compose up -d cipher-app
