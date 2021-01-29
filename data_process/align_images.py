'''
Use multiprocessing to deal with face, making aligned face.

@author: zz
@date: 2019.12.13
'''

from data_process.face_alignment import *
import multiprocessing


class aligner():
    def __init__(self, output_size=512, x_scale=1, y_scale=1, em_scale=0.1, use_alpha=False):
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.em_scale = em_scale
        self.use_alpha = use_alpha
        self.output_size = output_size
        cpus = 8 # multiprocessing.cpu_count()
        self.pools = multiprocessing.Pool(processes = cpus)
        self.results = []

    def align(self, raw_img_path, out_img_path, target_img_path=None):
        self.pools.apply_async(image_align, args=(raw_img_path, out_img_path, self.output_size, target_img_path))


    def get_result(self):
        return self.pools.get()

    def close(self):
        # 回收进程池
        self.pools.close()
        self.pools.join()
