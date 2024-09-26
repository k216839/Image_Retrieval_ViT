import os
import warnings
import time
import json
import configparser
import numpy as np
import cv2
import io
import redis
import uuid
import base64
import settings

from flask import Flask, request, jsonify, abort, render_template
from PIL import Image
from logger import AppLogger
from werkzeug.exceptions import HTTPException, default_exceptions
from datetime import date, datetime
from pytz import timezone, utc
from sklearn.metrics.pairwise import cosine_similarity
import torch

import utils
from model import VIT_MSN
warnings.filterwarnings("ignore")

time_zone = timezone('Asia/Ho_Chi_Minh')

# initialize Flask application
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__,
  static_folder=settings.STATIC_FOLDER,
  static_url_path="/static",
  template_folder=os.path.dirname(__file__)
)

logger = AppLogger()
config = configparser.ConfigParser(inline_comment_prefixes='#')
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

master_files, index = utils.load_config()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    json_response = {}
    return_code = 0

    if request.method == 'POST':
        # request time out
        time_out = 0
        # validate request
        if request.files.get('image') and settings.DEBUG:
            debug_image = request.files.get('image').read()
            debug_image = Image.open(io.BytesIO(debug_image)).convert('RGB')
            debug_image = np.array(debug_image)
            cv2.imwrite('debug.jpg', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
        
        start_time_rq = time.time()
        # Get file data
        if request.files.get(settings.FILE_KEY):
            time_out = time.time()
            img_raw_data = request.files.get(settings.FILE_KEY).read()
            image = Image.open(io.BytesIO(img_raw_data)).convert('RGB')
        elif request.form.get(settings.FILE_KEY):
            time_out = time.time()
            img_base64_str = request.form.get(settings.FILE_KEY)
            img_raw_data = base64.b64decode(img_base64_str)
            image = Image.open(io.BytesIO(img_raw_data))
        else:
            abort(400)

        # get nt_retr
        nr_retr = settings.NR_RETR
        if request.form.get(settings.NR_RETR_KEY) is not None:
            nr_retr = utils.try_int(request.form.get(settings.NR_RETR_KEY))
            if nr_retr > settings.MAX_NR_RETR:
                nr_retr = settings.MAX_NR_RETR

        # save request to cache db
        img_id = str(uuid.uuid4())
        logger.info('Request image with id: %s' % (img_id))

        # Save input
        if settings.CBIR_SAVE_QUERY_IMAGES:
            path_to_query_image = img_id + '.jpg'
            path_to_query_image = os.path.join(
                settings.QUERY_IMAGE_FOLDER, path_to_query_image)
            try:
                image.save(path_to_query_image)
            except IOError:
                logger.error(
                    "Cannot save the query image to file '%s'" % path_to_query_image)

        if settings.SEARCH_MODE == 1:
            # Create search request
            search_image = np.array(image)
            img_str = utils.base64_encode_image(search_image)

            req_obj = {settings.ID_KEY: img_id,
                        settings.IMAGE_WIDTH_KEY: image.width,
                        settings.IMAGE_HEIGHT_KEY: image.height,
                        settings.IMAGE_KEY: img_str,
                        }
            db.rpush(settings.IMAGE_QUEUE, json.dumps(req_obj))

            logger.info("Save %s image to %s success" %
                        (img_id, settings.IMAGE_QUEUE))

            # received searching & matching reponses
            state = 0  # 0 - searching | 1 - matching
            while True:
                run_time = int(round((time.time() - time_out) * 1000))
                if run_time >= settings.REQ_TIME_OUT:
                    logger.error("ALERT: Request %s time out" % img_id)
                    abort(408)

                # handler searching responses
                if state == 0:
                    output = db.get(img_id)
                    if output is not None:
                        db.delete(img_id)
                        output = output.decode("utf-8")
                        search_results = []
                        boxes_sod = []
                        boxes_sod.extend(json.loads(output)[settings.IMAGE_BOXES])

                        rt_categories = []
                        rt_categories.extend(json.loads(output)['category'])
                        # sorted search results list
                        search_results.extend(json.loads(output)['topn'])
                        logger.debug('searching: %s, %s' % (img_id, search_results))
                        search_results = sorted(
                            search_results, key=lambda x: x[settings.SCORE_KEY], reverse=True)
                        
                        if len(search_results) > 0:
                            json_response["type"] = "searching"
                            json_response['matched_files'] = search_results
                            if len(rt_categories) > 0:
                                json_response['category'] = rt_categories
                            json_response["status"] = 0
                            json_response["message"] = "successful"
                            return_code = 200
                        else:
                            json_response["status"] = 1
                            json_response["message"] = "No matches found"
                            return_code = 404
                        break
                # sleep 0.25s
                time.sleep(settings.SERVER_SLEEP)
        else:
            feature = model.get_features([image]).reshape(1, settings.VIT_FEATURE_LEN)
            distance_matrix = cosine_similarity(feature, index)
            distances, indices = torch.topk(torch.from_numpy(distance_matrix), k=nr_retr)
            search_results = []
            try:
                for (dis, ind) in zip(distances[0], indices[0]):
                    item = {
                        settings.IMAGE_KEY: master_files[ind],
                        settings.SCORE_KEY: dis.item()
                    }
                    search_results.append(item)
                json_response["type"] = "searching"
                json_response['matched_files'] = search_results
                json_response["status"] = 0
                json_response["message"] = "successful"
                return_code = 200
            except:
                json_response["status"] = 1
                json_response["message"] = "No matches found"
                return_code = 404

    # save statistic log
    msg = str(json_response["message"])
    if (msg == "successful"):
        arr = json_response["matched_files"]
        data = []
        for item in arr:
            data.append(item["image"])
        logger.info(
            f"statistic: {img_id}, {msg}, {'|'.join([str(elem) for elem in data])}")
    else:
        logger.info(f'statistic: {img_id}, {msg}, 0')
    end_time_rq = time.time()
    # send response
    result = json.dumps(json_response, ensure_ascii=False).encode("utf8")
    logger.info('response: %s, %s' % (img_id, result))
    logger.info(f'Total time process {img_id}: {end_time_rq - start_time_rq}')
    return result, return_code

@app.errorhandler(Exception)
def handle_exception(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    error_str = jsonify(message=str(e), code=code, success=False)
    logger.error("EXCEPTION: %s" % error_str.data.decode("utf-8"))
    return error_str

# register error handler
for ex in default_exceptions:
    app.register_error_handler(ex, handle_exception)

if __name__ == '__main__':
    model = VIT_MSN()
    # run server
    app.run(host="0.0.0.0", port=8000)
