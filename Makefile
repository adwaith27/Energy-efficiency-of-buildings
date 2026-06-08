.PHONY: train metrics grafana stop clean

train:
	uv run python train.py

metrics:
	uv run python metrics_server.py --port 8000 --refresh-seconds 30

grafana:
	docker compose up --build

stop:
	docker compose down

clean:
	rm -rf artifacts reports __pycache__ src/__pycache__
