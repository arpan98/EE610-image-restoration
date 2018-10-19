import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import cv2


def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def full_inverse():
    for channel in range(num_channels):
        G = np.fft.fftshift(np.fft.fft2(blurred_img[:, :, channel], (2*img_rows, 2*img_cols)))
        H = np.fft.fftshift(np.fft.fft2(kernel, (2*img_rows, 2*img_cols)))
        H = H / 255.0
        F_hat[:, :, channel] = G / H
        a = np.nan_to_num(F_hat[:, :, channel])
        f_hat[:, :, channel] = np.real(np.fft.ifft2(np.fft.ifftshift(a)))

    out_image = f_hat[:img_rows, :img_cols, :]
    out_image = np.abs(out_image)
    max_val = np.max(out_image)
    out_image = out_image / max_val
    out_image = (out_image * 3 * 255.0).astype(np.uint8)
    return out_image

def truncated_inverse(r):
    dist_matrix = np.zeros((2*img_rows, 2*img_cols))
    for i in range(2*img_rows):
        for j in range(2*img_cols):
            if ((i-img_rows)**2 + (j-img_cols)**2) < r**2:
                dist_matrix[i][j] = 1
    for channel in range(num_channels):
        G = np.fft.fftshift(np.fft.fft2(blurred_img[:, :, channel], (2*img_rows, 2*img_cols)))
        H = np.fft.fftshift(np.fft.fft2(kernel, (2*img_rows, 2*img_cols)))
        H = H / 255.0
        F_hat[:, :, channel] = G * dist_matrix / H
        a = np.nan_to_num(F_hat[:, :, channel])
        f_hat[:, :, channel] = np.real(np.fft.ifft2(np.fft.ifftshift(a)))

    out_image = f_hat[:img_rows, :img_cols, :]
    out_image = np.abs(out_image)
    max_val = np.max(out_image)
    out_image = out_image / max_val
    out_image = (out_image * 255.0).astype(np.uint8)
    return out_image

def wiener(K=100):
    for channel in range(num_channels):
        G = np.fft.fftshift(np.fft.fft2(blurred_img[:, :, channel], (2*img_rows, 2*img_cols)))
        H = np.fft.fftshift(np.fft.fft2(kernel, (2*img_rows, 2*img_cols)))
        H = H / 255.0
        F_hat[:, :, channel] = G * np.conj(H) / (np.abs(H)**2 + K)
        a = np.nan_to_num(F_hat[:, :, channel])
        f_hat[:, :, channel] = np.real(np.fft.ifft2(np.fft.ifftshift(a)))

    out_image = f_hat[:img_rows, :img_cols, :]
    out_image = np.abs(out_image)
    max_val = np.max(out_image)
    out_image = out_image / max_val
    out_image = (out_image * 255.0).astype(np.uint8)
    return out_image

def constrained_least_squares(gamma=3.7):
    p = np.array([[0, -1, 0], [-1, -4, -1], [0, -1, 0]])
    P = np.fft.fftshift(np.fft.fft2(p, (2*img_rows, 2*img_cols)))
    for channel in range(num_channels):
        G = np.fft.fftshift(np.fft.fft2(blurred_img[:, :, channel], (2*img_rows, 2*img_cols)))
        H = np.fft.fftshift(np.fft.fft2(kernel, (2*img_rows, 2*img_cols)))
        H = H / 255.0
        F_hat[:, :, channel] = G * np.conj(H) / (np.abs(H)**2 + gamma * (np.abs(P)**2))
        a = np.nan_to_num(F_hat[:, :, channel])
        f_hat[:, :, channel] = np.real(np.fft.ifft2(np.fft.ifftshift(a)))

    out_image = f_hat[:img_rows, :img_cols, :]
    out_image = np.abs(out_image)
    max_val = np.max(out_image)
    out_image = out_image / max_val
    out_image = (out_image * 255.0).astype(np.uint8)
    return out_image


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algo", help="full_inv/trunc_inv/wiener", type=str)
parser.add_argument("-p", "--parameter", help="parameter for different methods", type=float)
args = parser.parse_args()

original_img = cv2.imread("images/GroundTruth1_1_1.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

bgr_img = cv2.imread('images/Blurry1_1.jpg')
img_rows, img_cols, num_channels = bgr_img.shape
blurred_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

kernel = cv2.imread('Kernel1G_SingleTile.png', 0)
F_hat = np.zeros(shape=(2*img_rows, 2*img_cols, num_channels), dtype=np.complex64)
f_hat = np.zeros((2*img_rows, 2*img_cols, num_channels))

if args.algo == "full_inv":
    output = full_inverse()
elif args.algo == "trunc_inv":
    output = truncated_inverse(args.parameter)
elif args.algo == "wiener":
    output = wiener(args.parameter)
elif args.algo == "cls":
    output = constrained_least_squares(args.parameter)
else:
    output = truncated_inverse(200)
print(psnr(original_img, blurred_img))
print(psnr(original_img, output))
print(ssim(original_img, blurred_img, multichannel=True))
print(ssim(original_img, output, multichannel=True))
plt.subplot(131), plt.imshow(original_img)
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(blurred_img)
plt.title('Distorted image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(output)
plt.title('Restored image'), plt.xticks([]), plt.yticks([])
plt.show()
