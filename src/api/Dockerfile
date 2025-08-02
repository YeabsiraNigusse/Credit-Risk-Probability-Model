FROM python:3.10-slim

WORKDIR /app

# 1) Copy & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy all your API code
COPY main.py pydantic_models.py ./

# 3) Expose & run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
