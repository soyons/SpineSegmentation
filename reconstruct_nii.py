import nibabel as nib
import numpy as np
import cv2
import os


image_path = 'dataset/train/image'
class_path = 'dataset/train/classes'
model_pred_path = 'dataset/train/classes'
save_image_path = 'dataset/train/nii_reconstruct'

nii_files = os.listdir(image_path)
counter = 0
for nii_file in nii_files:
    counter += 1
    print('processing ' + str(counter) + ' of ' + str(len(nii_files)))
    image_name = image_path + '/' + nii_file
    img = nib.load(image_name)
    slices = img.get_data()
    (filename, extension) = os.path.splitext(nii_file)
    data = np.zeros(slices.shape, dtype=np.int16)
    for i in range(slices.shape[2]):
        pred_mask_name = model_pred_path + '/' + nii_file + '.' + str(i) + '.png'
        mask = np.int16(cv2.imread(pred_mask_name, cv2.IMREAD_GRAYSCALE))
        data[:, :, i] = mask

    reconstructed_nii = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    reconstructed_nii_name = save_image_path + '/reconstructed_' + nii_file + '.gz'
    nib.save(reconstructed_nii, reconstructed_nii_name)


