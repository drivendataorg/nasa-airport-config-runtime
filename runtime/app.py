import multiprocessing
import time

from flask import Flask
from loguru import logger
from werkzeug import run_simple
import src

app = Flask(__name__)


def server(queue: multiprocessing.Queue) -> None:
    @app.route("/shutdown")
    def shutdown():
        queue.put("shutdown")
        return "shutdown"

    @app.route("/health")
    def health():
        time.sleep(10)
        return "healthy"

    @app.route("/predict")
    def predict():
        time.sleep(3)
        return {"prediction": src.predict()}

    run_simple("localhost", 5001, app)


if __name__ == "__main__":
    # https://werkzeug.palletsprojects.com/en/2.0.x/serving/#shutting-down-the-server
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=server, args=(queue,))
    process.start()
    logger.info("Running")
    shutdown = queue.get(block=True)
    logger.info("Shutting down")
    process.terminate()
