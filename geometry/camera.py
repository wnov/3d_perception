import torch


def cam2img(points: torch.Tensor, intrinsics: torch.Tensor):
    """
    Project points from camera coordinate to img coordinate.

    Args:
        -ponits: coordinate in camera space, [num_points, 3(x, y, z)]
        -intrinsics: camera intrinsics, [3, 3]
    Return:
        -out: points coordinate in img space. [num_points, 3(u, v, d)]
    """

    num_points = points.shape[:-1]
    points = points.unsqueeze(-1)
    intrinsics = intrinsics.view(*([1] * len(num_points)), 3, 3).expand(
        *num_points, 3, 3
    )
    out = torch.matmul(intrinsics, points).squeeze(-1)
    out[..., :2] = out[..., :2] / out[..., 2:]
    return out


def img2cam(points: torch.Tensor, intrinsics: torch.Tensor):
    """
    Reproject points from img space to camera space.

    Args:
        -points: coordinate in img space, [num_points, 3(u, v, d)]
        -intrinsics: camera intrinsics, [3, 3]
    Return:
        -out: points coordinate in camera space. [num_points, 3(x, y)]
    """
    num_points = points.shape[:-1]
    points[..., :2] = points[..., :2] * points[..., 2:]
    points = points.unsqueeze(-1)
    intrinsics = torch.inverse(intrinsics)
    intrinsics = intrinsics.view(*([1] * len(num_points)), 3, 3).expand(
        *num_points, 3, 3
    )
    out = torch.matmul(intrinsics, points).squeeze(-1)
    return out


def world2cam(points: torch.Tensor, extrinsics: torch.Tensor):
    """
    Project points from world coordinate to camera coordinate.

    Args:
        -ponits: coordinate in worl space, [num_points, 3(x, y, z)]
        -extrinsics: camera extrinsics, world2camera, [3, 4]
    Return:
        -out: points coordinate in camera space. [num_points, 3(x, y, z)]
    """
    num_points = points.shape[:-1]
    points = points.unsqueeze(-1)
    extrinsics = extrinsics.view(*([1] * len(num_points)), 3, 4).expand(
        *num_points, 3, 4
    )

    rotation = extrinsics[..., :3]
    translation = extrinsics[..., 3:]
    points = torch.matmul(rotation, points)
    points += translation
    out = points.squeeze(-1)
    return out


def shift_intrinsics(delta: torch.Tensor, intrinsics: torch.Tensor):
    """
    Update intrinsics when shift img.
    Args:
        -delta: shift in img space, [2(d_u, d_v)].
        -intrinsics: camera intrinsics, [3, 3].
    Return:
        -out: updated camera intrinsics, [3, 3].
    """
    d_u, d_v = delta
    c_x, c_y = intrinsics[0, 2], intrinsics[1, 2]
    c_x += d_u
    c_y += d_v
    intrinsics[0, 2], intrinsics[1, 2] = c_x, c_y

    return intrinsics


def crop_intrinsics(delta: torch.Tensor, intrinsics: torch.Tensor):
    """
    Update intrinsics when crop img.
    Args:
        -delta: crop in img space, [2(d_u, d_v)].
        -intrinsics: camera intrinsics, [3, 3].
    Return:
        -out: updated camera intrinsics, [3, 3].
    """
    d_u, d_v = delta
    c_x, c_y = intrinsics[0, 2], intrinsics[1, 2]
    c_x -= d_u
    c_y -= d_v
    intrinsics[0, 2], intrinsics[1, 2] = c_x, c_y

    return intrinsics


def rotate_intrinsics(angle: float, intrinsics: torch.Tensor):
    """
    Update intrinsics when rotate img.
    Args:
        -angle: rotation in img space, float.
        -intrinsics: camera intrinsics, [3, 3].
    Return:
        -out: updated camera intrinsics, [3, 3].
    """
    rotation = torch.zeros(3, 3)
    rotation[0, 0] = torch.cos(rotation)
    rotation[0, 1] = torch.sin(-rotation)
    rotation[1, 0] = torch.sin(rotation)
    rotation[1, 1] = torch.cos(rotation)
    rotation[2, 2] = 1.0

    out = torch.matmul(rotation, intrinsics)
    return out


def resize_intrinsics(scale: torch.Tensor, intrinsics: torch.Tensor):
    """
    Update intrinsics when resize img.
    Args:
        -angle: factor in resize space, [2(scale_u, scale_v)].
        -intrinsics: camera intrinsics, [3, 3].
    Return:
        -intrinsics: updated camera intrinsics, [3, 3].
    """
    scale_x, scale_y = scale
    intrinsics[0] *= scale_x
    intrinsics[1] *= scale_y

    return intrinsics


if __name__ == "__main__":
    points = torch.randn(4, 3)
    intrinsics = torch.rand(3, 3)
    img_points = cam2img(points, intrinsics)
    cam_points = img2cam(img_points, intrinsics)
    print(torch.sum(cam_points - points) < 1e-4)
