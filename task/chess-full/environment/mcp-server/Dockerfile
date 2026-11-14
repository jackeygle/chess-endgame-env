FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir chess==1.11.2 fastmcp==4.2.0

WORKDIR /app
COPY chessenv/ chessenv/
COPY server.py positions.json oracle_token openings.json ./

EXPOSE 8000
CMD ["python", "server.py"]
