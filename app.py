import os
import zipfile
import shutil
from flask import Flask, render_template, request
import cv2
import numpy as np
import nibabel as nib
import xgboost as xgb

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def infer_multimodal(subject_folder, model_path="xgb_model.json"):
    subject_id = os.path.basename(subject_folder)
    flair_path = os.path.join(subject_folder, f"{subject_id}_flair.nii.gz")
    t1_path    = os.path.join(subject_folder, f"{subject_id}_t1.nii.gz")
    t1ce_path  = os.path.join(subject_folder, f"{subject_id}_t1ce.nii.gz")
    t2_path    = os.path.join(subject_folder, f"{subject_id}_t2.nii.gz")
    
    flair = nib.load(flair_path).get_fdata()
    t1    = nib.load(t1_path).get_fdata()
    t1ce  = nib.load(t1ce_path).get_fdata()
    t2    = nib.load(t2_path).get_fdata()
    
    slice_idx = flair.shape[2] // 2
    flair_slice = flair[:, :, slice_idx]
    t1_slice    = t1[:, :, slice_idx]
    t1ce_slice  = t1ce[:, :, slice_idx]
    t2_slice    = t2[:, :, slice_idx]
    
    def process_slice(img_slice):
        image_norm = cv2.normalize(img_slice, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image_uint8 = image_norm.astype(np.uint8)
        image_resized = cv2.resize(image_uint8, (224, 224)).astype(np.float32)
        return image_resized
    
    flair_processed = process_slice(flair_slice)
    t1_processed    = process_slice(t1_slice)
    t1ce_processed  = process_slice(t1ce_slice)
    t2_processed    = process_slice(t2_slice)
    
    feat = np.array([
        flair_processed.mean(), flair_processed.std(),
        t1_processed.mean(),    t1_processed.std(),
        t1ce_processed.mean(),  t1ce_processed.std(),
        t2_processed.mean(),    t2_processed.std()
    ]).reshape(1, -1)
    
    model = xgb.Booster()
    model.load_model(model_path)
    
    dmatrix = xgb.DMatrix(feat)
    pred = model.predict(dmatrix)
    return pred

def save_visuals(subject_folder, subject_id):
    flair_path = os.path.join(subject_folder, f"{subject_id}_flair.nii.gz")
    seg_path   = os.path.join(subject_folder, f"{subject_id}_seg.nii.gz")

    flair_data = nib.load(flair_path).get_fdata()
    slice_idx = flair_data.shape[2] // 2
    flair_slice = flair_data[:, :, slice_idx]
    flair_img = cv2.normalize(flair_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flair_img = cv2.resize(flair_img, (256, 256))

    flair_path_png = os.path.join(UPLOAD_FOLDER, f"{subject_id}_flair.png")
    cv2.imwrite(flair_path_png, flair_img)

    if os.path.exists(seg_path):
        seg_data = nib.load(seg_path).get_fdata()
        seg_slice = seg_data[:, :, slice_idx]
        seg_mask = np.ma.masked_where(seg_slice == 0, seg_slice)
        seg_overlay = cv2.applyColorMap((seg_mask.filled(0) * 50).astype(np.uint8), cv2.COLORMAP_JET)
        combined = cv2.addWeighted(flair_img, 0.7, seg_overlay, 0.5, 0)
        seg_path_png = os.path.join(UPLOAD_FOLDER, f"{subject_id}_seg.png")
        cv2.imwrite(seg_path_png, combined)
        return flair_path_png, seg_path_png

    return flair_path_png, None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    flair_img, seg_img = None, None

    if request.method == 'POST':
        file = request.files['zip_file']
        if file and file.filename.endswith('.zip'):
            zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_path = os.path.join(UPLOAD_FOLDER, os.path.splitext(file.filename)[0])
                if os.path.exists(extract_path):
                    shutil.rmtree(extract_path)
                zip_ref.extractall(extract_path)

            subject_id = os.listdir(extract_path)[0]
            subject_folder = os.path.join(extract_path, subject_id)

            pred_score = infer_multimodal(subject_folder)
            label = "Tumor" if pred_score[0] >= 0.5 else "No Tumor"
            result = f"{subject_id} => {label} (score: {pred_score[0]:.4f})"

            flair_img, seg_img = save_visuals(subject_folder, subject_id)

    return render_template("index.html", result=result, flair_img=flair_img, seg_img=seg_img)

if __name__ == '__main__':
    app.run(debug=False, threaded=False)
