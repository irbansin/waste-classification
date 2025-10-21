from PIL import Image

def divide_image_into_grid(pil_img, grid_size=3):
    w, h = pil_img.size
    patch_w, patch_h = w // grid_size, h // grid_size
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = i * patch_w
            upper = j * patch_h
            right = (i + 1) * patch_w if i < grid_size - 1 else w
            lower = (j + 1) * patch_h if j < grid_size - 1 else h
            patch = pil_img.crop((left, upper, right, lower))
            patches.append({
                "patch": patch,
                "coords": [int(left), int(upper), int(right), int(lower)]
            })
    return patches
