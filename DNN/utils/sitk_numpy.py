import numpy as np
import SimpleITK as sitk

__all__ = [
    "load_sitk_to_numpy",
    "save_numpy_to_sitk",
]


def load_sitk_to_numpy(path, header=False, dtype=None, dcm=False):
    if dcm:
        reader = sitk.ImageSeriesReader()
        dcm_series = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dcm_series)
        data = reader.Execute()
    else:
        data = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(data).swapaxes(0, 2)
    if dtype is not None:
        array = array.astype(dtype)
    if header:
        return {
            "array": array,
            "origin": np.array(data.GetOrigin()),
            "spacing": np.array(data.GetSpacing()),
            "direction": np.array(data.GetDirection()),
        }
    else:
        return {
            "array": array,
        }


def save_numpy_to_sitk(data, path, header=False):
    img = sitk.GetImageFromArray(data["array"].swapaxes(0, 2))
    if header:
        img.SetOrigin(data["origin"])
        img.SetSpacing(data["spacing"])
        img.SetDirection(data["direction"])
    sitk.WriteImage(img, path, True)
