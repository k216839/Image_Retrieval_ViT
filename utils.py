import sys
import base64
import numpy as np
import redis
import time

import settings
from logger import AppLogger

logger = AppLogger()

def try_int(value):
    try:
        return int(value)
    except:
        return -1

def base64_encode_image(a):
    return base64.b64encode(a).decode("utf-8")

def base64_decode_image_py37(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a

def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a


def multi_pop(r, q, n):
    arr = []
    count = 0
    while True:
        try:
            p = r.pipeline()
            p.multi()
            for i in range(n):
                p.lpop(q)
            arr = p.execute()
            return arr
        except redis.ConnectionError:
            count += 1
            logger.error("Connection failed in %s times" % count)
            if count > 3:
                raise
            backoff = count * 5
            logger.info('Retrying in {} seconds'.format(backoff))
            time.sleep(backoff)
            r = redis.StrictRedis(host=settings.REDIS_HOST,
                                  port=settings.REDIS_PORT, db=settings.REDIS_DB)
            
def load_config():
    # list image supports
    with open(settings.IMAGE_LIST_PATH, 'r') as img_file:
        candidate_ids = img_file.readlines()
    candidate_ids = [line.strip() for line in candidate_ids]
    # load pre-compute features
    index = np.load(settings.FEATURE_PATH)

    return candidate_ids, index