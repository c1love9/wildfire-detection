"""Isolated native camera reader. Only framed messages are written to stdout."""

import argparse
import json
import os
import struct
import sys
import time

import cv2


HEADER = struct.Struct('!cI')
FRAME = struct.Struct('!Qd')


def send(kind, body):
    sys.stdout.buffer.write(HEADER.pack(kind, len(body)) + body)
    sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--backend', choices=['AUTO', 'DSHOW', 'MSMF'], default='AUTO')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    backend = args.backend
    if backend == 'AUTO':
        backend = 'DSHOW' if os.name == 'nt' else 'AUTO'
    api = {'AUTO': cv2.CAP_ANY, 'DSHOW': cv2.CAP_DSHOW, 'MSMF': cv2.CAP_MSMF}[backend]
    capture = None
    try:
        capture = cv2.VideoCapture(args.index, api)
        if not capture.isOpened():
            raise RuntimeError('无法打开摄像头，请检查设备编号、其他软件占用和系统摄像头权限。')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        send(b'I', json.dumps({'backend': backend, 'fps': capture.get(cv2.CAP_PROP_FPS)}).encode())
        sequence = 0
        while True:
            ok, bgr = capture.read()
            stamp = time.monotonic()
            if not ok or bgr is None or not bgr.size:
                raise RuntimeError('摄像头没有返回画面，设备可能已断开或被其他软件占用。')
            factor = min(1, args.width / bgr.shape[1], args.height / bgr.shape[0])
            if factor < 1:
                bgr = cv2.resize(bgr, (max(2, round(bgr.shape[1] * factor)), max(2, round(bgr.shape[0] * factor))))
            ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError('摄像头画面传输编码失败。')
            sequence += 1
            send(b'F', FRAME.pack(sequence, stamp) + jpeg.tobytes())
    except (BrokenPipeError, OSError):
        pass  # Parent stopped this reader; native resources are released below.
    except Exception as exc:
        try:
            send(b'E', str(exc).encode('utf-8'))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if capture is not None:
            capture.release()


if __name__ == '__main__':
    main()
