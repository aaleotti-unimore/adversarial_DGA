import socket

if socket.gethostname() == "classificatoredga":
    import sys

    sys.path.append("../detect_DGA")

from features.features_testing import *
from model import *

logger = logging.getLogger(__name__)

hdlr = logging.FileHandler('run.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)

if __name__ == '__main__':
    model = cross_val(n_samples=100)
    # model = Model(directory="saved models/2017-09-14 13:56")
    compare(model)
    # cross_val()
    pass
