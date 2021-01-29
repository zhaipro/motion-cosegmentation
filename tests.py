from part_swap import *

# 初始化换脸器
checkpoint = "76-epc32.pth.tar" if True else "first-order-512-best.pth.tar"  # 模型文件路径
my_face_swaper = face_swaper(checkpoint=checkpoint, batch_size=8, device_ids=[0])

# 换脸
target_video = "driving_video_test.mp4"


source_image = "13.png"
result_video_name = "13_result3.mp4"
tmp_folder = "tmp_13"  # 中间文件存放文件夹
my_face_swaper.swap_face(source_image, target_video, tmp_folder, result_video_name=result_video_name)
