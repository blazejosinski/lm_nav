import h5py
import numpy as np
import os
import io
import utm
import cv2
import PIL.Image

def get_hdf5_leaf_names(node, name=''):
    if isinstance(node, str):
        assert os.path.exists(node)
        with h5py.File(node, 'r') as f:
            return get_hdf5_leaf_names(f)
    elif isinstance(node, h5py.Dataset):
        return [name]
    else:
        names = []
        for child_name, child in node.items():
            names += get_hdf5_leaf_names(child, name=name+'/'+child_name)
        return names

class Hdf5Loader(object):

    def __init__(self, hdf5_fname, compress=True):
        assert os.path.exists(hdf5_fname)
        self._hdf5_fname = hdf5_fname
        self._compress = compress

    def read(self):
        d = dict()
        with h5py.File(self._hdf5_fname, 'r') as f:
            keys = get_hdf5_leaf_names(f)
            for k in keys:
                v = np.array(f[k])
                d[k.lstrip('/')] = v
        return d

def im2bytes(arrs, format='jpg'):
    if len(arrs.shape) == 4:
        return np.array([im2bytes(arr_i, format=format) for arr_i in arrs])
    elif len(arrs.shape) == 3:
        im = PIL.Image.fromarray(arrs.astype(np.uint8))
        with io.BytesIO() as output:
            im.save(output, format="jpeg")
            return output.getvalue()
    else:
        raise ValueError


def bytes2im(arrs):
    if len(arrs.shape) == 1:
        return np.array([bytes2im(arr_i) for arr_i in arrs])
    elif len(arrs.shape) == 0:
        return np.array(PIL.Image.open(io.BytesIO(arrs)))
    else:
        raise ValueError


EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * np.pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0


def latlong_to_utm(latlong):
    """
    :param latlong: latlong or list of latlongs
    :return: utm (easting, northing)
    """
    if np.any(np.isnan(latlong)): return np.array([0., 0.])
    latlong = np.array(latlong)
    if len(latlong.shape) > 1:
        return np.array([latlong_to_utm(p) for p in latlong])

    easting, northing, _, _ = utm.from_latlon(*latlong)
    return np.array([easting, northing])


def utm_to_latlong(u, zone_number=10, zone_letter='S'):
    u = np.array(u)
    if len(u.shape) > 1:
        return np.array([utm_to_latlong(u_i, zone_number=zone_number, zone_letter=zone_letter) for u_i in u])


    easting, northing = u
    return utm.to_latlon(easting, northing, zone_number=zone_number, zone_letter=zone_letter)

def imrectify_fisheye(img, K, D, balance=0.0):
    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
    dim = img.shape[:2][::-1]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, dim, cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return undistorted_img


def imresize(image, shape, resize_method=PIL.Image.LANCZOS):
    if len(image.shape) == 4:
        return np.stack(
            [imresize(image_i, shape, resize_method=resize_method) for image_i in image]
        )

    assert len(shape) == 3
    assert shape[-1] == 1 or shape[-1] == 3
    assert (
        image.shape[0] / image.shape[1] == shape[0] / shape[1]
    )  # maintain aspect ratio
    height, width, channels = shape

    if len(image.shape) > 2 and image.shape[2] == 1:
        image = image[:, :, 0]

    im = PIL.Image.fromarray(image)
    im = im.resize((width, height), resize_method)
    im = np.array(im)

    if len(im.shape) == 2:
        im = np.expand_dims(im, 2)

    assert im.shape == tuple(shape)

    return im


def rectify_and_resize(image, shape, rectify=True):
    if rectify:
        ### jackal camera intrinsics
        fx, fy, cx, cy = 272.547000, 266.358000, 320.000000, 240.000000
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        D = np.array([[-0.038483, -0.010456, 0.003930, -0.001007]]).T
        balance = 0.5

        if len(image.shape) == 4:
            return np.array([rectify_and_resize(im_i, shape) for im_i in image])

        image = imrectify_fisheye(image, K, D, balance=balance)

    image = imresize(image, shape)

    return image


def crop_center(img):
    a = 96 + 112
    b = 16 + 112 - 30
    return img.crop((a, b, a + 224, b + 224))


def rectify_and_crop(img):
    np_image = np.array(img)
    np_image_rr = rectify_and_resize(np_image, np_image.shape)
    return crop_center(PIL.Image.fromarray(np_image_rr))