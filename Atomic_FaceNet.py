import json
import os
import time

from flask import Flask, request

from UseFaceNet import test_on_array, reload_records
from utils_FN import b64string2array

app = Flask(__name__)


@app.route('/imr-ai-service/atomic_functions/recognize', methods=['POST'])
def recognize():
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string = data['imgString']
        img = b64string2array(img_string)
        time_take = time.time()
        if img is None:
            result = []
        else:
            result = test_on_array(img)
        time_take = time.time() - time_take
        if "fileName" in data.keys():
            app.logger.info("recognition  return:{d},use time:{t}".format(d=result, t=time_take))
            return json.dumps({data['fileName']: [{'value': result}]}, ensure_ascii=False)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/reload', methods=['POST'])
def reload():
    if request.method == "POST":
        time_take = time.time()
        result = reload_records()
        time_take = time.time() - time_take
        return json.dumps(
            {
                'total': result[0],
                'new': result[1],
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


if __name__ == '__main__':
    pid = os.getpid()
    print('pid is:', pid)
    with open('atomic_pid.txt', 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("12246"),
        debug=False, threaded=True)
