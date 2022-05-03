import signal
import time


def running():
    signal.signal(signal.SIGINT, sigstop)
    signal.signal(signal.SIGTERM, sigstop)
    signal.signal(signal.SIGHUP, sigstop)
    while True:
        time.sleep(10)
        x = 1.0/0.0
        #raise Exception("THIS IS A RANDOM EXCEPTION")


def sigstop(a,b):
    print('Received:', a)
    print('PROGRAM TERMINATE')

if __name__ == "__main__":
    print("PROGRAM STARTS...")
    running()