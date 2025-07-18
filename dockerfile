# Use official lightweight Python image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py    # 👈 Add this line

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .
COPY xgb_baseline_model.pkl .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
