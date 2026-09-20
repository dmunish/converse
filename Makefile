PY ?= python
PORT ?= 8000

.PHONY: help install ingest run dev smoke clean

help:
	@echo "Converse — LearnForge support assistant"
	@echo ""
	@echo "  make install   Install Python deps"
	@echo "  make ingest    Build the LanceDB index from ./docs"
	@echo "  make run       Start the FastAPI backend on :$(PORT)"
	@echo "  make dev       Same, with autoreload"
	@echo "  make smoke     Hit /chat with a sample question"
	@echo "  make clean     Remove local DB / index / storage artifacts"

install:
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) ingest.py

run:
	uvicorn app:app --host 0.0.0.0 --port $(PORT)

dev:
	uvicorn app:app --reload --host 0.0.0.0 --port $(PORT)

smoke:
	curl -s -X POST http://localhost:$(PORT)/chat \
	  -H 'content-type: application/json' \
	  -d '{"message":"Can I get a refund for a course I bought 3 days ago?","session_id":"smoke"}' \
	  | $(PY) -m json.tool

clean:
	rm -rf ./lancedb ./storage ./converse.db