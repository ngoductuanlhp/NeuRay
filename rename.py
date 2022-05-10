import os
import shutil
import cv2

files = sorted(os.listdir('/home/ubuntu/nerf_ws/cpscan114_gt'))

for i, f in enumerate(files):
    img_path = os.path.join('/home/ubuntu/nerf_ws/cpscan114_gt', f)
    new_img_path = os.path.join('/home/ubuntu/nerf_ws/cpscan114_gt', f'{int(i)}.jpg')
    os.rename(img_path, new_img_path)


# image_dir = '/home/ubuntu/nerf_ws/cpscan114_ft/'
# new_image_dir = '/home/ubuntu/nerf_ws/cpscan114_ft_new/'
# os.makedirs(new_image_dir)

# files = sorted(os.listdir(image_dir))
# for i, f in enumerate(files):
#     img_path = os.path.join(image_dir, f)

#     img = cv2.imread(img_path)

#     img_new = cv2.resize(img, (640, 480))

#     cv2.imwrite(os.path.join(new_image_dir, f'{i}-nr_fine.jpg'), img_new)