import os
import pickle
import struct
from PIL import Image
import numpy as np

BYTES_PER_RECORD = 8199
ORIGINAL_SHAPE = (128, 127)
RESIZED_SHAPE = (28, 28)

file_list = ['ETL8G/ETL8G_{:02d}'.format(i) for i in range(1, 34)]

script_dir = os.path.dirname(os.path.abspath(__file__))
save_file = script_dir + '/etl8g.pkl'

label_types = [
    b'A.HIRA  ', b'I.HIRA  ', b'U.HIRA  ', b'E.HIRA  ', b'O.HIRA  ',
    b'KA.HIRA ', b'KI.HIRA ', b'KU.HIRA ', b'KE.HIRA ', b'KO.HIRA ',
    b'SA.HIRA ', b'SHI.HIRA', b'SU.HIRA ', b'SE.HIRA ', b'SO.HIRA ',
    b'TA.HIRA ', b'CHI.HIRA', b'TSU.HIRA', b'TE.HIRA ', b'TO.HIRA ',
    b'NA.HIRA ', b'NI.HIRA ', b'NU.HIRA ', b'NE.HIRA ', b'NO.HIRA ',
    b'HA.HIRA ', b'HI.HIRA ', b'FU.HIRA ', b'HE.HIRA ', b'HO.HIRA ',
    b'MA.HIRA ', b'MI.HIRA ', b'MU.HIRA ', b'ME.HIRA ', b'MO.HIRA ',
    b'YA.HIRA ', b'YU.HIRA ', b'YO.HIRA ',
    b'RA.HIRA ', b'RI.HIRA ', b'RU.HIRA ', b'RE.HIRA ', b'RO.HIRA ',
    b'WA.HIRA ', b'N.HIRA  '
]


def read_record(f):
    s = f.read(BYTES_PER_RECORD)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    pixels = r[-1]
    label_type = r[2]
    return (pixels, label_type)


def load_img(buffer):
    img = Image.frombytes('F', ORIGINAL_SHAPE, buffer, 'bit', 4)
    img = img.convert('L')
    return img


def load_dataset_from_file(file_path):
    X = []
    y = []
    size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        record_count = 0
        while (record_count * BYTES_PER_RECORD) < size:
            f.seek(record_count * BYTES_PER_RECORD)
            img_pixels, label_type = read_record(f)

            # ひらがなだけを抽出
            if label_type in label_types:
                img = load_img(img_pixels)
                img = img.resize(RESIZED_SHAPE)
                X.append(np.array(img).ravel())

                label = label_types.index(label_type)
                y.append(label)

            record_count += 1

    return (X, y)


def one_hot(y):
    label_size = np.unique(y).size
    T = np.zeros([y.size, label_size])
    for i, row in enumerate(T):
        row[y[i]] = 1

    return T


def init_dataset():
    X = []
    y = []
    for path in file_list:
        _X, _y = load_dataset_from_file(path)
        X.extend(_X)
        y.extend(_y)

    X = np.array(X)
    y = np.array(y)

    with open(save_file, 'wb') as f:
        pickle.dump((X, y), f, -1)


def load_dataset():
    if not os.path.exists(save_file):
        init_dataset()

    with open(save_file, 'rb') as f:
        X, y = pickle.load(f)

    y = one_hot(y)

    return (X, y)


if __name__ == '__main__':
    X, y = load_dataset()
    print(X.shape)
    print(y.shape)
