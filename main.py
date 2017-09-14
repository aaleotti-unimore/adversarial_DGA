import sys
import logging

sys.path.append("../detect_DGA")

from model import Model, compare, cross_val

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr = logging.FileHandler('logging.log')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)

if __name__ == '__main__':
    model = cross_val()
    # model = Model(directory="saved models/2017-09-14 19:36")
    compare(model)
    pass
