import tempfile
import tarfile
import tensorflow as tf
import numpy as np
import boto3
import matplotlib.pyplot as plt
import os
from PIL import Image

from osgeo import gdal
import sys

print(tf.__version__)

sys.path.append('../functions') 
#from functions import download_model_from_s3
from functions.smooth_tiled_predictions import predict_img_with_smooth_windowing

def download_model_from_s3(bucket_name, model_key, download_path):
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, model_key, download_path)

MODEL_S3_PTH = 's3://weo-dl-models/tree_monitor/semseg/tm-semseg-lu-orthophoto/tm-semseg-lu-ortho-2023-NM-v00/models/tm-semseg-lu-ortho-2023-NM-v00-2024-08-28-05-22-38-445/output/model.tar.gz'
#INFERENCE_IMG_S3_PTH = 's3://weo-clients/naturpark-mellerdall/orthophoto/2023/steinheim_orchards_30cm_small.tif'
MODEL_S3_BUCKET, MODEL_S3_PREFIX = MODEL_S3_PTH.split("s3://")[1].split("/",1)
# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TEMP_MODEL_DIR = tempfile.mkdtemp()
TEMP_MODEL_PTH = os.path.join(TEMP_MODEL_DIR, os.path.basename(MODEL_S3_PTH))
S3 = boto3.client('s3')
print(TEMP_MODEL_DIR)
print(TEMP_MODEL_PTH)

# Download the extracted model tar.gz file from S3
try:
    download_model_from_s3(bucket_name=MODEL_S3_BUCKET, model_key=MODEL_S3_PREFIX, download_path=TEMP_MODEL_PTH)
except Exception as e:
    print(e)
# Open the tar.gz file and extract the contents
with tarfile.open(TEMP_MODEL_PTH, 'r:gz') as tar:
    tar.extractall(path=TEMP_MODEL_DIR)
try:
    model = tf.saved_model.load(os.path.join(TEMP_MODEL_DIR, 'unet', '1'))
except Exception as e:
    print(e)

dset = gdal.Open('naturpark_orthophoto_2023_1m.tif')
width=dset.RasterXSize
height=dset.RasterYSize
print(width,height)
tilesize=1280
for i in range(0,width,tilesize):
    for j in range(0,height,tilesize):
         w = min(i+tilesize, width) - i
         h = min(j+tilesize, height) - j
         gdaltranString = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " +str(h)+" " + "naturpark_orthophoto_2023_1m.tif" + " " + "naturpark_orthophoto_2023_1m_translate" + "_"+str(i)+"_"+str(j)+".tif"
         os.system(gdaltranString)
         print("reading array")
         ds = gdal.Open("naturpark_orthophoto_2023_1m_translate" + "_"+str(i)+"_"+str(j)+".tif")
         arr_i_j = ds.ReadAsArray()[:3]
         arr_i_j = arr_i_j.transpose(1,2,0)
         print(w,h)
         predictions_smooth = predict_img_with_smooth_windowing(
              arr_i_j.astype(np.float32),
              window_size=128,
              subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
              nb_classes=2,
              pred_func=(
                   lambda img_batch_subdiv: model(img_batch_subdiv)
              ))
         print(type(predictions_smooth[:,:,1]))
         outputImage=f"naturpark_orthophoto_2023_1m_" + "_"+str(i)+"_"+str(j)+".tif"
         image = Image.fromarray(predictions_smooth[:,:,1])
         image.save(outputImage, format='TIFF')
         ds=None
         os.remove(f"naturpark_orthophoto_2023_1m_translate" + "_"+str(i)+"_"+str(j)+".tif")

         