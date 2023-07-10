import os
import cv2
import numpy as np
import paddle

from natsort import natsorted
import config
import imgproc
from model_VDSR import VDSR
import os


model_path='pd_model/model.pdparams'
os.environ['KMP_DUPLICATE_LIB_OK'] ='TRUE'

def main() :
    # Initialize the super-resolution model
    model = VDSR()
    print("Build VDSR model successfully.")

    # Load the super-resolution model weights

    layer_state_dict=paddle.load(model_path)
    model.set_state_dict(layer_state_dict)

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()

    # Initialize the image evaluation index.
    total_psnr = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.hr_dir))#自然排序
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(hr_image_path)}`...")
        # Make high-resolution image
        hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.0
        hr_image_height, hr_image_width = hr_image.shape[:2]
        hr_image_height_remainder = hr_image_height % 12
        hr_image_width_remainder = hr_image_width % 12
        hr_image = hr_image[:hr_image_height - hr_image_height_remainder, :hr_image_width - hr_image_width_remainder, ...]

        # Make low-resolution image
        lr_image = cv2.resize(hr_image,None,fx=1 / config.upscale_factor,fy=1 / config.upscale_factor,interpolation=cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, None, fx=config.upscale_factor, fy=config.upscale_factor,interpolation=cv2.INTER_CUBIC)
        # Convert BGR image to YCbCr image
        lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=False)

        # Split YCbCr image data
        lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
        hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert Y image data convert to Y tensor data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=True).unsqueeze_(0).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=True).unsqueeze_(0).unsqueeze_(0)

        # Only reconstruct the Y channel image data.通过网络
        with paddle.no_grad():
            #sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)
            sr_y_tensor = paddle.clip(model(lr_y_tensor),0,1.0)

        # Cal PSNR
        total_psnr += 10. * paddle.log10(1. / paddle.mean((sr_y_tensor - hr_y_tensor) ** 2))

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)

    #print(f"PSNR: {total_psnr / total_files:4.2f}dB.\n")
    print(total_psnr.numpy()/total_files)


if __name__ == "__main__":
    main()
