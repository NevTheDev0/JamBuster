import click
import time
from multiprocessing import Process
from traffic_logger import start_logger
from sim_RT_prediction import make_predict


@click.group()
def jambuster():
    pass


@jambuster.command()
def run():
    logger_proc = Process(target=start_logger)
    predict_proc = Process(target=make_predict)

    logger_proc.start()
    time.sleep(10)
    predict_proc.start()

    logger_proc.join()
    predict_proc.join()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    jambuster()
