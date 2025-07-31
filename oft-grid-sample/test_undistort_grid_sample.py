import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from loguru import logger
import pytest


def undistort_image_pytorch(img, cx, cy, fx, fy, k1):
    """
    img: Tensor of shape (1, H, W) or (C, H, W), values in [0, 1]
    cx, cy: principal point
    fx, fy: focal lengths
    k1: single radial distortion coefficient
    Returns: undistorted image, same shape as input
    """
    C, H, W = img.shape if img.ndim == 3 else (1,) + img.shape
    # Generate normalized undistorted grid
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    xs = xs.float()
    ys = ys.float()
    # Map to normalized camera coordinates
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    r2 = x**2 + y**2
    # Apply distortion (forward)
    coeff = 1 + k1 * r2
    x_dist = x * coeff
    y_dist = y * coeff
    # Map back to pixel coordinates
    u_dist = x_dist * fx + cx
    v_dist = y_dist * fy + cy
    # Normalize to [-1, 1] for grid_sample
    grid_x = 2 * (u_dist / (W - 1)) - 1
    grid_y = 2 * (v_dist / (H - 1)) - 1
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid = grid.unsqueeze(0)  # (1, H, W, 2)
    if img.ndim == 2:
        img = img.unsqueeze(0)
    else:
        img = img
    img = img.unsqueeze(0)  # (1, C, H, W)

    # grid_sample expects (N, C, H, W) and (N, H, W, 2)
    logger.info(f"img.shape: {img.shape}, grid.shape: {grid.shape}")
    undistorted = F.grid_sample(img, grid, align_corners=True, mode="bilinear", padding_mode="zeros")

    # Remove batch dimension
    return undistorted.squeeze(0)  # (C, H, W)


def save_and_plot_images(output_file: str, original_img, undistorted_img, undistorted_img_params: str):
    # Plot the original and undistorted images side by side
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img.permute(1, 2, 0).numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(undistorted_img.permute(1, 2, 0).numpy())
    axes[1].set_title(f"Undistorted Image\n{undistorted_img_params}")
    axes[1].axis("off")

    # Save the plot to disk
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


@pytest.mark.parametrize(
    "image_path",
    [
        "/home/mbezulj/lena.png",
    ],
)
def test_undistort_image(image_path):
    # Load a real image
    img = Image.open(image_path)

    # Transform the image to a tensor
    img = transforms.ToTensor()(img)

    H, W = img.shape[1], img.shape[2]

    # Mock camera intrinsics
    cx, cy = W / 2 + 10, H / 2 + 100
    fx, fy = W / 2 + 55, H / 2 + 55
    k1 = 0.25

    undistorted = undistort_image_pytorch(img, cx, cy, fx, fy, k1)
    output_file = f"undistorted_{image_path.split('/')[-1]}"
    save_and_plot_images(output_file, img, undistorted, f"(cx={cx}, cy={cy}, fx={fx}, fy={fy}, k1={k1})")
