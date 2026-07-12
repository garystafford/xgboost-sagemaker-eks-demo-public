FROM public.ecr.aws/docker/library/python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir xgboost==1.7.6 fastapi uvicorn numpy

COPY app.py ./
COPY xgboost-model ./

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
