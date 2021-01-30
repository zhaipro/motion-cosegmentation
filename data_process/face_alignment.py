import numpy as np
import scipy.ndimage
import os
import bz2, math
import cv2 as cv
import PIL.Image
import dlib
from pathlib import Path
root_path = Path(__file__).parent

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        self.threshold = 0

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets, scores, idx = self.detector.run(img, 1, 0)  # The seconde param always be 1, which means upsample the image 1 time,
                                                            # this will make everything bigger and allow us to detect more faces.
                                                            # The third param is score.

        for i, detection in enumerate(dets):
            try:
                if scores[i] < self.threshold:
                    continue
                # print(f'image: {image}   i:{i}   score: {scores[i]}')
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                yield face_landmarks
            except:
                print("Exception in get_landmarks()!")


landmarks_model_path = os.path.join(root_path, 'shape_predictor_68_face_landmarks.dat')
landmarks_detector = LandmarksDetector(landmarks_model_path)


def get_dist(A, B):
    return ((A[0]-B[0])**2+(A[1]-B[1])**2)**(1/2)


def image_align(src_file, dst_file, output_size, trg_file, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
        """
        src_file: required，需要换的脸的图片路径
        dst_file: required，新图片保存的路径
        trg_file: optional，原视频中的脸的第一帧图片路径，如果给出来就要根据这张脸对source image作对齐
        """
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        face_landmarks = landmarks_detector.get_landmarks(src_file)

        # Get all faces
        lms = list(face_landmarks)

        # 只要第一张脸
        lm = lms[0]
        lm = np.array(lm)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= x_scale
        y = np.flipud(x) * [-y_scale, y_scale]
        c = eye_avg + eye_to_mouth * em_scale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = np.uint8(np.clip(np.rint(img), 0, 255))
            if alpha:
                mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
                mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
                img = np.concatenate((img, mask), axis=2)
                img = PIL.Image.fromarray(img, 'RGBA')
            else:
                img = PIL.Image.fromarray(img, 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # 规范生成图片的后缀名，统一设为'.jpg'.
        dst_file_this = dst_file.replace('.jpeg', '.jpg')
        dst_file_this = dst_file.replace('.png', '.jpg')
        img.convert('RGB').save(dst_file_this, 'JPEG')


        # 根据target图片作对齐
        if trg_file:
            # 获取图片
            src_img = cv.imread(dst_file_this)

            # 获取脸部关键点
            src_lmks = landmarks_detector.get_landmarks(dst_file_this)
            trg_lmks = landmarks_detector.get_landmarks(trg_file)

            # 只要图片中的第一张脸
            src_lm = list(src_lmks)[0]
            src_lm_left  = src_lm[0]
            src_lm_right = src_lm[16]
            trg_lm = list(trg_lmks)[0]
            trg_lm_left  = trg_lm[0]
            trg_lm_right = trg_lm[16]

            # 计算缩放大小
            src_dist = get_dist(src_lm_left, src_lm_right)
            trg_dist = get_dist(trg_lm_left, trg_lm_right)
            dist_scale = trg_dist/src_dist
            src_shape_new = (int(src_img.shape[1]*dist_scale), int(src_img.shape[0]*dist_scale))
            scale_src_img = cv.resize(src_img, src_shape_new)

            # 计算旋转角度
            src_dy = src_lm_right[1] - src_lm_left[1]
            src_dx = src_lm_right[0] - src_lm_left[0]
            src_angle = math.atan2(src_dy, src_dx) * 180. / math.pi
            trg_dy = trg_lm_right[1] - trg_lm_left[1]
            trg_dx = trg_lm_right[0] - trg_lm_left[0]
            trg_angle = math.atan2(trg_dy, trg_dx) * 180. / math.pi
            angle = trg_angle - src_angle
            rotate_matrix = cv.getRotationMatrix2D(src_lm_left, angle, scale=1)
            rotate_src_img = cv.warpAffine(scale_src_img, rotate_matrix, src_shape_new)

            # 计算平移
            dy = trg_lm_left[1]-src_lm_left[1]*dist_scale
            dx = trg_lm_left[0]-src_lm_left[0]*dist_scale
            M = np.float32([[1,0,dx], [0,1,dy]])
            move_src_img = cv.warpAffine(rotate_src_img, M, src_shape_new)

            # crop原画布大小
            new_src_img = move_src_img[:src_img.shape[0],:src_img.shape[1]]
            cv.imwrite(dst_file_this, new_src_img)
