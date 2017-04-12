import argparse
import os
import struct

from PIL import Image

_ORIGINAL_BYTES_PER_RECORD = 8199
_ORIGINAL_SHAPE = (128, 127)
_IMAGE_SIZE = 28

_LABEL_TYPES = [
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


def _read_record(f):
    s = f.read(_ORIGINAL_BYTES_PER_RECORD)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    pixels_buffer = r[-1]
    label_type = r[2]
    return (pixels_buffer, label_type)


def _load_img(buffer):
    img = Image.frombytes('F', _ORIGINAL_SHAPE, buffer, 'bit', 4)
    img = img.convert('L')
    return img


def _convert(f, size):
    label_pixel_records = bytes()

    record_count = 0
    while (record_count * _ORIGINAL_BYTES_PER_RECORD) < size:
        f.seek(record_count * _ORIGINAL_BYTES_PER_RECORD)
        pixels_buffer, label_type = _read_record(f)

        if label_type in _LABEL_TYPES:
            img = _load_img(pixels_buffer)
            img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE))

            label = _LABEL_TYPES.index(label_type)

            fmt = '>B{0}s'.format(_IMAGE_SIZE ** 2)
            label_pixel_records += struct.pack(fmt, label, img.tobytes())

        record_count += 1

    return label_pixel_records


def main(data_dir):
    file_paths = [os.path.join(data_dir, 'ETL8G_{:02d}'.format(i))
                  for i in range(1, 34)]

    for fp in file_paths:
        if not os.path.exists(fp):
            raise ValueError('Failed to find file: {0}'.format(fp))

    train_file_paths = file_paths[:26]
    test_file_paths = file_paths[26:]

    curr_dir = os.path.dirname(__file__)
    train_save_file = os.path.join(curr_dir, 'etlcdb.bin')
    test_save_file = os.path.join(curr_dir, 'etlcdb_test.bin')

    def _process_files(paths, save_to):
        with open(save_to, 'wb') as save_file:
            for p in paths:
                size = os.path.getsize(p)
                with open(p, 'rb') as f:
                    buf = _convert(f, size)
                    save_file.write(buf)

    _process_files(train_file_paths, train_save_file)
    _process_files(test_file_paths, test_save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, default='ETL8G',
        help='Directory path to ETL8G data')
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)

    main(data_dir)
