.PHONY: install ingest dry cli api ui eval ablation test traces refused cost all

install:      ## install dependencies
	pip install -r requirements.txt

ingest:       ## build the index (needs OPENAI_API_KEY)
	python ingest.py

dry:          ## extract + chunk only, no API calls, no key needed
	python ingest.py --dry-run

cli:          ## ask a question in the terminal
	python cli.py

api:          ## serve POST /query on :8000
	uvicorn app.api:app --reload --port 8000

ui:           ## Streamlit reviewer UI on :8501
	streamlit run app/ui.py

eval:         ## run the 21-question eval, regenerate eval/qa_log.md
	python eval/run_eval.py

ablation:     ## measure BM25 vs dense vs hybrid vs hybrid+rerank
	python eval/ablation.py

test:         ## unit tests - no API key required
	pytest tests/ -q

traces:       ## list recent query traces
	python tools/traces.py

refused:      ## list only the queries that were refused, and by which gate
	python tools/traces.py --refused

cost:         ## token spend per stage across all traced queries
	python tools/traces.py --cost

all: install ingest test eval
