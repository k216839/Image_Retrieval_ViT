# initialize Redis connection settings
REDIS_HOST = "0.0.0.0"
REDIS_PORT = 6379
REDIS_DB = 0
WEB_REDIS_HOST = "192.168.1.70"
WEB_REDIS_PORT = 6379
WEB_REDIS_DB = 0
WEB_SERVER_HOST = "http://192.168.1.70"
SERVER_HOST = 'http://35.198.217.246:8000'


# data type
VIT_FEATURE_LEN = 384
MTC_IMAGE_WIDTH = 700
MTC_IMAGE_HEIGHT = 700
MTC_IMAGE_CHANS = 3
SEARCH_MODE = 0

# initialize constants used for server queuing
BATCH_SIZE = 1
NR_RETR =10
MAX_NR_RETR = 20

SERVER_SLEEP = 0.05
CLIENT_SLEEP = 0.05

# initialize content keys
FILE_KEY = "file"
ID_KEY = "id"
IMAGE_KEY = "image"
IMAGE_WIDTH_KEY = "width"
IMAGE_HEIGHT_KEY = "height"
SCORE_KEY = "score"
IMG_LIST_FILE_KEY = 'candidate_ids'
DATABASE_KEY = 'DATABASE'
IMAGE_QUEUE = 'image_queue'
IMAGE_LIST_PATH = 'mmt_features/candidate_ids.txt'
FEATURE_PATH = 'mmt_features/features.npy'
NR_RETR_KEY = 'nr_retr'
QUERY_IMAGE_FOLDER = 'query_images'
DB_IMAGE_FOLDER = 'data/oxbuild/images'
# logger info
LOGGER_INFO_PATH = "logs/info"
LOGGER_ERROR_PATH = "logs/error"
LOGGER_CONF_NAME = "logger.conf"

# save query images
CBIR_SAVE_QUERY_IMAGES = True
DEBUG = True
# requests timeout
REQ_TIME_OUT = 5000 #ms
APP_DB_NAME = 'oxbuild_database'

STATIC_FOLDER = ''
