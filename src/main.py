#!/opt/venv/bin/python

"""Main routine"""
import sys
import logging
from datetime import datetime

import package as pkg


if __name__ == '__main__':

    logfile = f'../log{datetime.now().strftime("%d%m%Y")}'
    logging.basicConfig(filename=logfile, level=logging.DEBUG, format="%(asctime)s %(message)s")

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('==============================================')
    logging.info('Template Repo: Program started at %s', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    p = pkg.Package()
    p.say_hi()
