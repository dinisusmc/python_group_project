install:
	pip install -r requirements.txt

api:
	cd models && uvicorn api:app --reload

app:
	cd models && python3 app.py