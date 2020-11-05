import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from Base.config import img_size
from cfg_FN import record_path, vector_path
from utils_FN import array2b64string, process_request

hot_start = False


def crop_image(raw_img):
    b64string = array2b64string(raw_img).decode()
    try:
        raise ConnectionError
        detect_result = process_request('fd_dbf', req_dict={'imgString': b64string})
        x1 = detect_result['res'][0][0]
        y1 = detect_result['res'][0][1]
        x2 = detect_result['res'][0][2]
        y2 = detect_result['res'][0][3]
    except Exception as e:
        print(repr(e))
        landscape = raw_img.shape[0] < raw_img.shape[1]
        if landscape:
            x_c = int(raw_img.shape[1] / 2)
            h = raw_img.shape[0]
            x1 = x_c - int(h / 2)
            y1 = 0
            x2 = x_c + int(h / 2)
            y2 = h
        else:
            y_c = int(raw_img.shape[0] / 2)
            w = raw_img.shape[1]
            x1 = 0
            y1 = y_c - int(w / 2)
            x2 = w
            y2 = y_c + int(w / 2)
    w = x2 - x1
    h = y2 - y1
    edge_size = max(w, h)
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    x1 = int(x_c - (edge_size / 2))
    x2 = int(x_c + (edge_size / 2))
    y1 = int(y_c - (edge_size / 2))
    y2 = int(y_c + (edge_size / 2))
    raw_img = cv2.resize(raw_img[y1:y2, x1:x2], (img_size, img_size))
    return raw_img


img = cv2.imread('test.png')
img = crop_image(img)


# cv2.imshow('c', img)
# cv2.waitKey()


def activate_gpu():
    gpu_list = tf.config.experimental.list_physical_devices(device_type="GPU")
    print(gpu_list)
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)


activate_gpu()
model = load_model('models/1in1out.h5')
with tf.device("/cpu:0"):
    result = model.predict(np.expand_dims(img, axis=0))


def get_vector(input_img):
    input_img = crop_image(input_img)
    with tf.device("/cpu:0"):
        output_result = model.predict(np.expand_dims(input_img, axis=0))
    return output_result


name_list = []
vector_list = []


def purge_vectors(purge_target=vector_path):
    file_list = os.listdir(purge_target)
    for file in file_list:
        full_name = os.path.join(purge_target, file)
        if os.path.exists(full_name):
            os.remove(full_name)
    file_list = os.listdir(purge_target)
    assert len(file_list) == 0


def incremental_vectorization():
    vector_file_list = os.listdir(vector_path)
    img_list = os.listdir(record_path)
    for index, file_name in enumerate(img_list):
        valid = file_name.endswith('.png')
        valid = valid or file_name.endswith('.jpg')
        valid = valid or file_name.endswith('.PNG')
        valid = valid or file_name.endswith('.JPG')
        if valid:
            if file_name + '.npy' in vector_file_list:
                print(file_name + '.npy', '已存在于特征库中')
            else:
                print(file_name + '.npy', '不存在于特征库中')
                src_img = cv2.imread(os.path.join(record_path, file_name))
                vector = get_vector(src_img)
                np.save(file=os.path.join(vector_path, file_name + '.npy'), arr=vector)
                print(file_name + '.npy', '新增特征向量完毕')


def reload_records(read_npy=True):
    count_before = len(name_list)
    name_list.clear()
    vector_list.clear()
    if read_npy:
        print('热启动')
        incremental_vectorization()
        file_list = os.listdir(vector_path)
        for index, file_name in enumerate(file_list):
            print('读取第', index, '张头像，名称为', file_name)
            start = datetime.datetime.now()
            name_list.append(file_name.replace('.npy', ''))
            vector = np.load(file=os.path.join(vector_path, file_name))
            print('特征向量已提取。')
            vector_list.append(vector)
            end = datetime.datetime.now()
            print('耗时', str(end - start))
    else:
        print('冷启动')
        purge_vectors()
        file_list = os.listdir(record_path)
        for index, file_name in enumerate(file_list):
            print('读取第', index, '张头像，名称为', file_name)
            start = datetime.datetime.now()
            valid = file_name.endswith('.png')
            valid = valid or file_name.endswith('.jpg')
            valid = valid or file_name.endswith('.PNG')
            valid = valid or file_name.endswith('.JPG')
            if valid:
                name_list.append(file_name)
                src_img = cv2.imread(os.path.join(record_path, file_name))
                vector = get_vector(src_img)
                np.save(file=os.path.join(vector_path, file_name + '.npy'), arr=vector)
                print('特征向量已提取。')
                vector_list.append(vector)
            end = datetime.datetime.now()
            print('耗时', str(end - start))
    count_after = len(name_list)
    return count_after, count_after - count_before


reload_records(read_npy=hot_start)


def calculate_distance(vector_a, vector_b):
    distance = np.linalg.norm(vector_a - vector_b)
    return distance


def test_on_array(input_img):
    input_vector = get_vector(input_img)
    distance_list = []
    for index, vector in enumerate(vector_list):
        distance = calculate_distance(vector, input_vector)
        distance_list.append(distance)
    distance = float(min(distance_list))
    index = distance_list.index(distance)
    name = name_list[index]
    return name, distance


if __name__ == '__main__':
    pass
