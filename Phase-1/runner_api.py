from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

def run(cmd: list[str]):
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"status": "ok", "stdout": out.stdout[-4000:], "stderr": out.stderr[-4000:]}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=(e.stderr or str(e))[-4000:])

@app.post("/run/weather")
def weather():
    return run(["python", "scripts/ingest_weather_latest.py"])

@app.post("/run/grid")
def grid():
    return run(["python", "scripts/ingest_eia_latest.py"])

@app.post("/run/price")
def price():
    return run(["python", "scripts/ingest_nyiso_price_latest.py"])

@app.get("/health")
def health():
    return {"status": "alive"}
