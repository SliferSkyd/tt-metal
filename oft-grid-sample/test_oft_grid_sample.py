import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from loguru import logger
import pytest
import matplotlib.pyplot as plt
import os


def get_oft_grids(feature_size=(1, 256, 48, 160)):
    ref_file = f"bbox_corners_{feature_size[0]}_{feature_size[1]}_{feature_size[2]}_{feature_size[3]}.pt"
    import os

    if os.path.exists(ref_file):
        bbox_corners = torch.load(ref_file)
    else:
        assert False, f"Reference file {ref_file} does not exist. Please run the script to generate it."

    # bbox_corners.shape torch.Size([1, 7, 25281, 4])
    top_left_grid = bbox_corners[..., [0, 1]].squeeze(0)  # 7, 25281, 2
    btm_right_grid = bbox_corners[..., [2, 3]].squeeze(0)  # 7, 25281, 2
    top_left_grid = bbox_corners[..., [2, 1]].squeeze(0)  # 7, 25281, 2
    btm_left_grid = bbox_corners[..., [0, 3]].squeeze(0)  # 7, 25281, 2

    return [top_left_grid, btm_right_grid, top_left_grid, btm_left_grid]


def get_area_and_visibility(img_height, img_width, features):
    assert img_height == features[2], f"Image height {img_height} does not match feature height {features[2]}"
    assert img_width == features[3], f"Image width {img_width} does not match feature width {features[3]}"

    ref_file = f"bbox_corners_{features[0]}_{features[1]}_{features[2]}_{features[3]}.pt"
    import os

    if os.path.exists(ref_file):
        bbox_corners = torch.load(ref_file)
    else:
        assert False, f"Reference file {ref_file} does not exist. Please run the script to generate it."

    EPSILON = 1e-6
    area = (
        (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + EPSILON
    ).unsqueeze(1)
    visible = area > EPSILON

    return area, visible


def save_and_plot(output_file_base_name, feature_size, original_img, grids):
    output_filename = f"{feature_size[0]}_{feature_size[1]}_{feature_size[2]}_{feature_size[3]}_{output_file_base_name}"

    area, visible = get_area_and_visibility(original_img.shape[1], original_img.shape[2], feature_size)
    points = area.numel()
    visible_points = visible.sum().item()
    area_and_visibility_msg = (
        f"Points in area: {points}, Visible points: {visible_points}; ratio = {visible_points/points:.2f}"
    )
    logger.info(area_and_visibility_msg)

    num_of_grids = len(grids)
    vertical_slices = grids[0].shape[0]

    og = original_img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(num_of_grids, vertical_slices, figsize=(10 * vertical_slices, 5 * num_of_grids))
    fig.suptitle(f"{output_file_base_name} {feature_size} {area_and_visibility_msg}", fontsize=14)

    for i, grid_point in enumerate(grids):
        # for vertical slice
        for v in range(grid_point.shape[0]):
            axes[i][v].imshow(og)
            axes[i][v].set_title(f"Grid {i} - Vertical {v}")
            axes[i][v].axis("off")

            x = grid_point[v, :, 0].squeeze(0)
            y = grid_point[v, :, 1].squeeze(0)

            # Convert normalized coordinates to pixel coordinates
            img_height, img_width = original_img.shape[1], original_img.shape[2]
            # 0.5 adjust for the center of the pixel
            x = (x + 1) / 2 * img_width - 0.5
            y = (y + 1) / 2 * img_height - 0.5

            axes[i][v].scatter(x, y, color="red", marker=".", s=2)

    # Save the plot to disk
    plt.tight_layout()
    plt.savefig(f"grid_{output_filename}.png")
    plt.close()

    # plot per vertical slice all grid points in one view
    fig, axes = plt.subplots(1, vertical_slices, figsize=(20 * vertical_slices, 10))
    fig.suptitle(f"{output_file_base_name} {feature_size} {area_and_visibility_msg}", fontsize=14)

    colors = ["red", "blue", "green", "yellow"]
    for v in range(grid_point.shape[0]):
        for i, grid_point in enumerate(grids):
            axes[v].set_title(f"Vertical {v}")

            x = grid_point[v, :, 0].squeeze(0)
            y = grid_point[v, :, 1].squeeze(0)

            axes[v].scatter(x, y, color=colors[i], marker=".", s=2)
            axes[v].legend([f"Grid {i}" for i in range(len(grids))], loc="upper right")

    # Save the plot to disk
    plt.tight_layout()
    plt.savefig(f"grid_raw_{output_filename}.png")
    plt.close()


@pytest.mark.parametrize("image_path", ["../cityscapes/leftImg8bit/val/munster/munster_000001_000019_leftImg8bit.png"])
@pytest.mark.parametrize("feature_size", [(1, 256, 48, 160), (1, 256, 24, 80), (1, 256, 12, 40)])
def test_oft_grid_sample(image_path, feature_size):
    img = Image.open(image_path)

    _, _, H, W = feature_size

    # Resize the image to match the feature size
    img = img.resize((W, H), Image.BILINEAR)

    # Transform the image to a tensor
    img = transforms.ToTensor()(img)  # (3, H, W)

    grids = get_oft_grids(feature_size=feature_size)

    save_and_plot(os.path.basename(image_path).split(".")[0], feature_size, img, grids)
