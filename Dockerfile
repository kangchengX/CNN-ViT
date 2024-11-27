FROM python:3.10

WORKDIR /app

COPY . /app

# Support system-level library libGL.so.1
RUN apt-get update && apt-get install -y libgl1

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]

CMD ["--help"]
