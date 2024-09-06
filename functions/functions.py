import logging
import multiprocessing
import os
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from glob import glob
from pathlib import Path
from urllib.parse import urlparse

import click
from osgeo import gdal, ogr, osr
from shapely.wkt import loads

from build_vrt import gdal_build_vrt_rastersource
from co_registration.apply_arrosics import (
    apply_align,
    apply_align_extend,
    apply_align_extend_file,
    apply_align_with_border,
)
from geoutils.utils import (
    change_projection,
    check_projection,
    find_file_by_tile_reference,
    get_tile_references,
)
from pyDMS import run as run_pydms
from sentinel2.operations.co_registration_operation import CoRegistrationOperation
from storage.aws import download as download_cache
from storage.aws import download_directory, upload_directory

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

if "PROJ_LIB" not in os.environ:
    os.environ["PROJ_LIB"] = r"/venv/share/proj/"

DEFAULT_NO_DATA_VALUE = -99999.0
DEFAULT_RUN_BUCKET = "airflow-runs"
DOWNLOAD_RELATIVE_PATH = "download"
TARGET_RELATIVE_PATH = "target"
CO_REG_RELATIVE_PATH = "co_reg"
ALIGNED_RELATIVE_PATH = "aligned"
SHARPEN_RELATIVE_PATH = "sharpen"
MERGED_RELATIVE_PATH = "merged"
COMPOSITE_RELATIVE_PATH = "composite"
TILED_RELATIVE_PATH = "tiled"
TILED_HIGH_RES_RELATIVE_PATH = "tiled_high_res"
TILE_SIZE_DEFAULT = 256
TILE_10M_SIZE = 512
TILE_OVERLAP = 10
TILE_10M_OVERLAP = 20
RESOLUTION_PRECISION = 10


BANDS_DICT = {
    "10": ["B02", "B03", "B04", "B08"],
    "20": ["B05", "B06", "B07", "B11", "B12"],
}

SOURCE_BANDS_RES_GLOB = {
    "10": (
        "*B02*.tif",
        "*B03*.tif",
        "*B04*.tif",
        "*B08*.tif",
    ),
    "20": (
        "*B05*.tif",
        "*B06*.tif",
        "*B07*.tif",
        "*B11*.tif",
        "*B12*.tif",
        "*B8A*.tif",
    ),
}


@click.command()
@click.option("--run_id")
@click.option("--work_bucket")
@click.option("--work_bucket_prefix")
@click.option("--area", default=None, help="Area to process")
@click.option("--reference_picture_remote_path", default=None)
@click.option("--reference_picture_for_high_res_remote_path", default=None)
@click.option(
    "--high_res_remote_path",
    default=None,
    help="Remote s3 location of the High res image. If Specified the and sharpening is given bands are sharpened with this image. Specify the resolutions to sharpen",
)
@click.option("--coreg_min_reliability", default=60, type=int)
@click.option("--high_res_resample_res", default=0, type=int)
@click.option("--target_srs", default="32631")
@click.option("--file_prefix", default="WEO")
@click.option("--tile_size", default=256)
@click.option("--tile_overlap", default=10)
@click.option("--pydms_moving_window", default=128)
def __process_random_folder(
    run_id,
    work_bucket,
    work_bucket_prefix,
    area,
    reference_picture_remote_path,
    reference_picture_for_high_res_remote_path,
    high_res_remote_path,
    coreg_min_reliability,
    high_res_resample_res,
    target_srs,
    file_prefix,
    tile_size,
    tile_overlap,
    pydms_moving_window,
):
    process_random_folder(
        run_id,
        work_bucket,
        work_bucket_prefix,
        area,
        reference_picture_remote_path,
        reference_picture_for_high_res_remote_path,
        high_res_remote_path,
        coreg_min_reliability=coreg_min_reliability,
        high_res_resample_res=high_res_resample_res,
        target_srs=target_srs,
        file_prefix=file_prefix,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        pydms_moving_window=pydms_moving_window,
    )
    return


def process_random_folder(
    run_id,
    work_bucket,
    work_bucket_prefix,
    area,
    reference_picture_remote_path,
    reference_picture_for_high_res_remote_path,
    high_res_remote_path,
    coreg_min_reliability=60,
    high_res_resample_res=0,
    work_path="./work",
    target_srs="32631",
    file_prefix="WEO",
    tile_size=256,
    tile_overlap=10,
    pydms_moving_window=128,
):
    sharpen_res = [10, 20]
    logging.info(f"area:{area}")
    logging.info(f"work_bucket:{work_bucket}")
    logging.info(f"work_bucket_prefix:{work_bucket_prefix}")
    logging.info(f"work_path:{work_path}")
    logging.info(f"target_srs:{target_srs}")
    logging.info(f"reference_picture_remote_path:{reference_picture_remote_path}")
    logging.info(
        f"reference_picture_for_high_res_remote_path:{reference_picture_for_high_res_remote_path}"
    )
    logging.info(f"high_res_remote_path:{high_res_remote_path}")
    logging.info(f"sharpen_res:{sharpen_res}")
    logging.info(f"file_prefix:{file_prefix}")
    logging.info(f"run_id:{run_id}")
    logging.info(f"coreg_min_reliability:{coreg_min_reliability}")
    logging.info(f"high_res_resample_res:{high_res_resample_res}")
    logging.info(f"pydms_moving_window:{pydms_moving_window}")

    # New Airflow Version submits string None
    reference_picture_remote_path = (
        None
        if reference_picture_remote_path == "None"
        else reference_picture_remote_path
    )
    reference_picture_for_high_res_remote_path = (
        None
        if reference_picture_for_high_res_remote_path == "None"
        else reference_picture_for_high_res_remote_path
    )
    high_res_remote_path = (
        None if high_res_remote_path == "None" else high_res_remote_path
    )

    # Preparation
    start_time = time.time()

    num_cpus = multiprocessing.cpu_count()
    logging.info(f"CPU Counts {num_cpus}")
    # Upload result if run_id given

    upload_result = run_id is not None or run_id != ""
    download_path = os.path.join(work_path, DOWNLOAD_RELATIVE_PATH)
    reference_download_path = os.path.join(work_path, "reference_data")
    target_path = os.path.join(work_path, TARGET_RELATIVE_PATH)
    merged_path = os.path.join(work_path, MERGED_RELATIVE_PATH)
    composite_path = os.path.join(work_path, COMPOSITE_RELATIVE_PATH)
    tiled_path = os.path.join(work_path, TILED_RELATIVE_PATH)
    tiled_high_res_path = os.path.join(work_path, TILED_HIGH_RES_RELATIVE_PATH)
    aligned_path = os.path.join(work_path, ALIGNED_RELATIVE_PATH)
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(reference_download_path, exist_ok=True)
    if os.path.exists(tiled_path) and os.path.isdir(tiled_path):
        shutil.rmtree(tiled_path)
    os.makedirs(tiled_path, exist_ok=True)
    os.makedirs(composite_path, exist_ok=True)
    if os.path.exists(tiled_high_res_path) and os.path.isdir(tiled_high_res_path):
        shutil.rmtree(tiled_high_res_path)
    os.makedirs(tiled_high_res_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    co_reg_path = os.path.join(work_path, CO_REG_RELATIVE_PATH)
    sharpen_path = os.path.join(work_path, SHARPEN_RELATIVE_PATH)
    target_srs_ref = osr.SpatialReference()
    target_srs_ref.ImportFromEPSG(int(target_srs))
    # 0. Download Assets from bucket
    download_directory(
        bucket_name=work_bucket,
        remote_folder_name=work_bucket_prefix,
        target_path=target_path,
        include_subdirectories=False,
    )

    file_to_process_list = glob(target_path + "/**/*.tif", recursive=True)
    logging.info(f"Files to process found: [{file_to_process_list}]")
    files_current_location = {}
    for f in file_to_process_list:
        files_current_location[Path(f).stem] = f
        change_projection(
            f,
            f,
            target_srs=target_srs_ref,
        )

    # 0.A Download filename if not yet on disk
    logging.info(f"{round(time.time() - start_time)} seconds")
    # 0.B Download reference file (if given) (REFERENCE_FILE_URL) to persistent storage
    reference_picture_target_local_path = None
    if (
        reference_picture_remote_path is not None
        and urlparse(reference_picture_remote_path).scheme == "s3"
    ):
        reference_picture_target_local_path = os.path.join(
            reference_download_path,
            urlparse(reference_picture_remote_path).path.split("/")[-1],
        )
        if not os.path.isfile(reference_picture_target_local_path):
            logging.info("Downloading reference_picture from s3...")
            t=urlparse(reference_picture_remote_path).path.strip("/")
            q=reference_picture_target_local_path
            r=urlparse(reference_picture_remote_path).netloc
            logging.info(t)
            logging.info(q)
            logging.info(r)
            download_cache(
                urlparse(reference_picture_remote_path).path.strip("/"),
                reference_picture_target_local_path,
                urlparse(reference_picture_remote_path).netloc,
            )
            logging.info(f"{round(time.time() - start_time)} seconds")
        else:
            logging.info(
                f"Skipping download of Reference file as it is already in {reference_download_path} / {reference_picture_target_local_path}"
            )
    # 1. Unzip package
    previous_op_path = target_path
    previous_res_path = {10: target_path, 20: target_path}
    # ALIGN
    if not os.path.exists(aligned_path) and not os.path.isdir(aligned_path):
        os.makedirs(aligned_path, exist_ok=True)
        source_file_list = files_current_location.values()
        for file in source_file_list:
            files_current_location[Path(file).stem] = shutil.copy(file, aligned_path)

        if reference_picture_target_local_path is not None:
            source_file_list = files_current_location.values()
            logging.info(f"Source Files identified For Align: {source_file_list}")
            reference_picture_target_local_path = shutil.copy(
                reference_picture_target_local_path, aligned_path
            )
            if area is not None:
                logging.info("Cropping Reference Picture to Area")
                geo_area = ogr.CreateGeometryFromWkt(area)
                minX, maxX, minY, maxY = geo_area.GetEnvelope()
                reference_picture_aligned_local_path = os.path.join(
                    os.path.split(os.path.abspath(reference_picture_target_local_path))[
                        0
                    ],
                    os.path.splitext(
                        os.path.basename(reference_picture_target_local_path)
                    )[0]
                    + "_al.tif",
                )
                change_projection(
                    reference_picture_target_local_path,
                    reference_picture_target_local_path,
                    target_srs=target_srs_ref,
                )
                apply_align_extend_file(
                    minX=minX,
                    minY=minY,
                    maxX=maxX,
                    maxY=maxY,
                    file=reference_picture_target_local_path,
                    out_file=reference_picture_aligned_local_path,
                    extend_srs="EPSG:4326",
                )

                reference_picture_target_local_path = (
                    reference_picture_aligned_local_path
                )
            apply_align(
                reference_picture_target_local_path, source_file_list, target_srs_ref
            )
            for aligned_file in glob(aligned_path + "/**/*.tif", recursive=True):
                filename = Path(aligned_file).stem.replace("_al", "")
                logging.info(f"is file {filename} in {files_current_location.keys()}")
                if filename in files_current_location.keys():
                    files_current_location[filename] = aligned_file
        else:
            if area is not None or area != "":
                source_file_list = glob(aligned_path + "/*.tif", recursive=True)
                logging.info(
                    f"Source Files identified For Align: {source_file_list} without reference picture but using AREA"
                )
                try:
                    logging.info(
                        f"Cropping Files to Area  {source_file_list} and [{area}]"
                    )
                    geo_area = loads(area)
                    minX, maxX, minY, maxY = ogr.CreateGeometryFromWkt(
                        str(geo_area)
                    ).GetEnvelope()

                    apply_align_extend(
                        minX=minX,
                        minY=minY,
                        maxX=maxX,
                        maxY=maxY,
                        file_list=source_file_list,
                        extend_srs="EPSG:4326",
                        target_srs=target_srs_ref,
                    )
                    ## Only use aligned fiels with _al ending
                    for aligned_file in glob(
                        aligned_path + "/**/*_al.tif", recursive=True
                    ):
                        filename = Path(aligned_file).stem.replace("_al", "")
                        logging.info(
                            f"is file {filename} in {files_current_location.keys()}"
                        )
                        if filename in files_current_location.keys():
                            files_current_location[filename] = aligned_file
                except:
                    logging.error("Problem")
        logging.info("Current File location -> ")
        logging.info(files_current_location)
        logging.info(f"{round(time.time() - start_time)} seconds")

    if reference_picture_target_local_path is not None:
        # 4. Calculate GRID_RES (if asked by params) for co_reg_local
        grid_res_calculated = None
        logging.info(f"Grid resolution calculated: {grid_res_calculated}")
        resolutions = {}
        for f in files_current_location.values():
            file_gdal = gdal.Open(f)
            geo_transform = file_gdal.GetGeoTransform()
            file_res = geo_transform[1]
            rounded_resolution = str(round(file_res, RESOLUTION_PRECISION))
            logging.info(f"Found {rounded_resolution} for {f}")
            if rounded_resolution in resolutions:
                l = resolutions[rounded_resolution]
                l.append(f)
                resolutions[rounded_resolution] = l
            else:
                resolutions[rounded_resolution] = [f]
        # CO_REG
        for resolution in resolutions.keys():
            logging.info(f"Starting CO_REG for {resolution}")
            # 1. Run CO_REG for each resolution
            source_file_list = resolutions[resolution]
            if len(source_file_list) > 0:
                logging.info(
                    f"Processing files {source_file_list} with {reference_picture_target_local_path}"
                )
                current_co_reg_path = os.path.join(co_reg_path, f"R{resolution}m")
                coreg_op = CoRegistrationOperation(
                    source_files=source_file_list,
                    target_path=current_co_reg_path,
                    pxl_resolution=resolution,
                    reference_file_path=reference_picture_target_local_path,
                    point_res_pixels=grid_res_calculated,
                    min_reliability=coreg_min_reliability,
                )
                coreg_op.run()
                for coreg_file in glob(
                    current_co_reg_path + "/**/*.tif", recursive=True
                ):
                    filename = Path(coreg_file).stem.replace("_al_aro", "")
                    logging.info(
                        f"is file {filename} in {files_current_location.keys()}"
                    )
                    if filename in files_current_location.keys():
                        files_current_location[filename] = coreg_file

        logging.info("Current File location -> ")
        logging.info(files_current_location)
        logging.info(f"{round(time.time() - start_time)} seconds")
        previous_op_path = co_reg_path

    if not os.path.exists(sharpen_path) or not os.path.isdir(sharpen_path):
        logging.info("Sharpening started...")
        # Clean follow up folders
        if os.path.exists(merged_path) and os.path.isdir(merged_path):
            shutil.rmtree(merged_path)
        os.makedirs(sharpen_path, exist_ok=True)
        if high_res_remote_path is not None and high_res_remote_path != "":
            logging.info("Starting Band Sharpening against High Resolution Image")
            high_res_picture_download_local_path = None
            if high_res_remote_path is not None:
                if urlparse(high_res_remote_path).scheme == "s3":
                    high_res_picture_download_local_path = os.path.join(
                        reference_download_path,
                        urlparse(high_res_remote_path).path.split("/")[-1],
                    )
                    if not Path(high_res_picture_download_local_path).is_file():
                        logging.info("Downloading high_res_file from s3...")
                        download_cache(
                            urlparse(high_res_remote_path).path.strip("/"),
                            high_res_picture_download_local_path,
                            urlparse(high_res_remote_path).netloc,
                        )
                else:
                    logging.info("high_res file is local")
                    high_res_picture_download_local_path = high_res_remote_path
                # Make sure high_res image has correct SRS
                logging.info("Cropping High res Picture to Input Layer")
                high_res_picture_aligned_local_path_with_file = os.path.join(
                    os.path.split(
                        os.path.abspath(high_res_picture_download_local_path)
                    )[0],
                    os.path.splitext(
                        os.path.basename(high_res_picture_download_local_path)
                    )[0]
                    + "_al_with_file.tif",
                )
                change_projection(
                    high_res_picture_download_local_path,
                    high_res_picture_download_local_path,
                    target_srs=target_srs_ref,
                )
                logging.info(
                    f"Found file to apply align {list(files_current_location.values())[0]}"
                )
                apply_align(
                    reference_file=list(files_current_location.values())[0],
                    source_file_list=[high_res_picture_download_local_path],
                    target_file_path=high_res_picture_aligned_local_path_with_file,
                    target_align_pixels=True,
                )

                if (
                    reference_picture_for_high_res_remote_path is not None
                    and reference_picture_for_high_res_remote_path != ""
                ):
                    reference_picture_for_high_res_local_path = os.path.join(
                        reference_download_path,
                        urlparse(reference_picture_for_high_res_remote_path).path.split(
                            "/"
                        )[-1],
                    )
                    if not Path(reference_picture_for_high_res_local_path).is_file():
                        logging.info(
                            "Downloading high_res_file reference file  from s3..."
                        )
                        download_cache(
                            urlparse(
                                reference_picture_for_high_res_remote_path
                            ).path.strip("/"),
                            reference_picture_for_high_res_local_path,
                            urlparse(reference_picture_for_high_res_remote_path).netloc,
                        )
                    # Make sure high_res image has correct SRS
                    logging.info("Cropping High res Picture to Area")
                    change_projection(
                        reference_picture_for_high_res_local_path,
                        reference_picture_for_high_res_local_path,
                        target_srs=target_srs_ref,
                    )
                    geo_area = ogr.CreateGeometryFromWkt(area)
                    minX, maxX, minY, maxY = geo_area.GetEnvelope()
                    reference_picture_for_high_res_local_path_al = os.path.join(
                        os.path.split(
                            os.path.abspath(reference_picture_for_high_res_local_path)
                        )[0],
                        os.path.splitext(
                            os.path.basename(reference_picture_for_high_res_local_path)
                        )[0]
                        + "_al.tif",
                    )
                    logging.info(
                        f"File -> {reference_picture_for_high_res_local_path_al}"
                    )
                    apply_align_extend_file(
                        file=reference_picture_for_high_res_local_path,
                        out_file=reference_picture_for_high_res_local_path_al,
                        minX=minX,
                        minY=minY,
                        maxX=maxX,
                        maxY=maxY,
                        extend_srs="EPSG:4326",
                        target_srs=target_srs_ref,
                    )
                    logging.info("Starting CoREG for high_res file")
                    high_res_co_reg_path = os.path.join(co_reg_path, "high_res")

                    gdal_high_res = gdal.Open(
                        high_res_picture_aligned_local_path_with_file
                    )
                    geo_data = gdal_high_res.GetGeoTransform()
                    high_res_resample_res = round(geo_data[1], RESOLUTION_PRECISION)
                    gdal_high_res = None

                    gdal_high_res_ref = gdal.Open(
                        reference_picture_for_high_res_local_path_al
                    )
                    geo_data_ref = gdal_high_res_ref.GetGeoTransform()
                    high_res_res_ref = round(geo_data_ref[1], RESOLUTION_PRECISION)
                    gdal_high_res_ref = None

                    logging.info(
                        f"Coreg resolution: [{high_res_resample_res}] vs original [{round(geo_data[1], RESOLUTION_PRECISION)}] vs ref [{high_res_res_ref}]"
                    )
                    logging.info(
                        f"File {high_res_picture_aligned_local_path_with_file} coregeed with {reference_picture_for_high_res_local_path_al}"
                    )

                    coreg_op = CoRegistrationOperation(
                        source_files=[high_res_picture_aligned_local_path_with_file],
                        target_path=high_res_co_reg_path,
                        pxl_resolution=high_res_res_ref,
                        reference_file_path=reference_picture_for_high_res_local_path_al,
                        min_reliability=coreg_min_reliability,
                    )
                    coreg_op.run()
                    high_res_picture_aligned_local_path_with_file = glob(
                        high_res_co_reg_path + "/**/*.tif", recursive=True
                    )[0]
                # Do Coreg for High_res picture
                logging.info("High_rs picture ready")
                bands_path = os.path.join(work_path, "bands_extracted")
                os.makedirs(bands_path, exist_ok=True)
                for current_file in files_current_location.values():
                    current_file_folder = os.path.join(
                        sharpen_path, Path(current_file).stem
                    )
                    os.makedirs(current_file_folder, exist_ok=True)
                    current_bands_folder = os.path.join(
                        bands_path, Path(current_file).stem
                    )
                    os.makedirs(current_bands_folder, exist_ok=True)
                    # Extract Bands separately

                    current_gdal = gdal.Open(current_file)
                    logging.info(
                        f"RasterCount for {current_file} is {current_gdal.RasterCount}"
                    )
                    for band_idx in range(current_gdal.RasterCount):
                        logging.info(f"Extracting Band {band_idx}")
                        # get the band object
                        band = current_gdal.GetRasterBand(band_idx + 1)

                        # create the output file name
                        output_file = os.path.join(
                            current_bands_folder,
                            Path(current_file).stem
                            + "_band"
                            + str(band_idx + 1)
                            + ".tif",
                        )

                        # create the output dataset
                        driver = gdal.GetDriverByName("GTiff")
                        out_ds = driver.Create(
                            output_file,
                            current_gdal.RasterXSize,
                            current_gdal.RasterYSize,
                            1,
                            band.DataType,
                        )

                        # set the geotransform and projection information
                        out_ds.SetGeoTransform(current_gdal.GetGeoTransform())
                        out_ds.SetProjection(current_gdal.GetProjection())

                        # write the band data to the output file
                        out_ds.GetRasterBand(1).WriteArray(band.ReadAsArray())
                        out_ds.FlushCache()
                        # close the output file
                        out_ds = None

                        current_band = Path(output_file)
                        logging.info(
                            f"Working on band for {current_file} and found band file {current_band}"
                        )
                        current_band_file_folder = os.path.join(
                            sharpen_path, current_band.stem
                        )
                        os.makedirs(current_band_file_folder, exist_ok=True)
                        current_band_gdal = gdal.Open(str(current_band))
                        geo_data = current_band_gdal.GetGeoTransform()
                        res = round(geo_data[1], RESOLUTION_PRECISION)

                        logging.info(
                            f"Sharpening File {str(current_band)} with Resolution [{res}]"
                        )
                        high_res_tiled_source = os.path.join(
                            tiled_high_res_path, str(res)
                        )
                        if not os.path.exists(
                            high_res_tiled_source
                        ) or not os.path.isdir(high_res_tiled_source):
                            os.makedirs(high_res_tiled_source, exist_ok=True)
                            gdal_high_res = gdal.Open(
                                high_res_picture_aligned_local_path_with_file
                            )
                            geo_data = gdal_high_res.GetGeoTransform()
                            pixel_size_high_res_picture = geo_data[1]
                            logging.info(
                                f"Found pixel size for high_res [{pixel_size_high_res_picture}]"
                            )
                            high_res_picture_tile_size = int(
                                round(res / pixel_size_high_res_picture) * tile_size
                            )

                            high_res_picture_tile_overlap = int(
                                round(res / pixel_size_high_res_picture) * tile_overlap
                            )

                            logging.info(
                                f"Calculated Tile size for high_res [{high_res_picture_tile_size}], tiling {high_res_picture_aligned_local_path_with_file} now..."
                            )
                            tile_cmd = f"gdal_retile.py -ps {high_res_picture_tile_size} {high_res_picture_tile_size} -overlap {high_res_picture_tile_overlap} -targetDir {high_res_tiled_source} {high_res_picture_aligned_local_path_with_file}"
                            logging.info(tile_cmd)
                            os.system(tile_cmd)
                        else:
                            logging.info(
                                "Skipping tiles generatinon as is already here"
                            )
                        high_res_tiles_file_list = []
                        high_res_tiles_file_list.extend(
                            glob(
                                high_res_tiled_source + "/*.tif",
                                recursive=True,
                            )
                        )
                        logging.info(
                            f"Checking files in [{high_res_tiled_source}/*.tif] "
                        )
                        logging.info(
                            f"Found HIGH_RES  tiles {len(high_res_tiles_file_list)} "
                        )
                        aligned_tiled_path = os.path.join(
                            os.path.join(work_path, "algined_tiles"), current_band.stem
                        )
                        os.makedirs(aligned_tiled_path, exist_ok=True)
                        #
                        futures = []
                        logging.info(f"Found  tiles {len(high_res_tiles_file_list)} ")
                        with ProcessPoolExecutor(num_cpus - 1) as e:
                            for high_res_tile in high_res_tiles_file_list:
                                tileX, tileY = get_tile_references(high_res_tile)
                                cloudmask_tile = ""
                                # Cut out area of low_res_band
                                aligned_tile = os.path.join(
                                    aligned_tiled_path,
                                    Path(high_res_tile).stem + ".tif",
                                )
                                apply_align_with_border(
                                    high_res_tile,
                                    str(current_band),
                                    None,
                                    aligned_tile,
                                    True,
                                    False,
                                    0,
                                )

                                logging.info(
                                    f"Running pyDMS on file {aligned_tile} with {high_res_tile} with cloudmask [{cloudmask_tile}]"
                                )
                                current_out_file_path = str(
                                    os.path.join(
                                        current_band_file_folder,
                                        os.path.splitext(
                                            os.path.basename(str(current_band))
                                        )[0]
                                        + "_"
                                        + str(tileX)
                                        + "_"
                                        + str(tileY)
                                        + "_sharpened.tif",
                                    )
                                )
                                futures.append(
                                    e.submit(
                                        run_pydms.run,
                                        high_res_tile,
                                        aligned_tile,
                                        current_out_file_path,
                                        cloudmask_tile,
                                        True,
                                        [255],
                                        pydms_moving_window,
                                        False,
                                    )
                                )
                            logging.info(
                                f"Waiting for [{len(futures)}] futures to finish"
                            )
                            for future in as_completed(futures):
                                task_result = future.result()
                                logging.info(f"Task {task_result} finished")
                            done_futures, not_done_futures = wait(
                                futures, timeout=None, return_when="ALL_COMPLETED"
                            )
                            logging.info(
                                f"Done waiting DONE [{len(done_futures)}], NOT_DONE [{len(not_done_futures)}]"
                            )

                        p = Path(f"{current_band_file_folder}")
                        sharpened_tiles = [
                            str(x)
                            for x in p.glob("*.tif")
                            if x.name.find("residual") < 0
                        ]
                        logging.info(
                            f"Found sharpened tiles to merge in path [{p}] and number [{len(sharpened_tiles)}]"
                        )
                        os.makedirs(merged_path, exist_ok=True)
                        current_band_file_merge_path = os.path.join(
                            merged_path, current_band.stem
                        )
                        os.makedirs(current_band_file_merge_path, exist_ok=True)
                        if len(sharpened_tiles) > 0:
                            logging.info(
                                f"Found files [{len(sharpened_tiles)}] for band"
                            )
                            merge_dst_path = os.path.join(
                                current_band_file_merge_path,
                                f"{file_prefix}_{current_band.stem}.tif",
                            )
                            cmd_merge = f"gdal_merge.py -n nan -ot Float32 -of GTiff -o {merge_dst_path} {' '.join(sharpened_tiles)}"
                            logging.info(cmd_merge)
                            os.system(cmd_merge)

                            # Check that all bands are uploaded
                            logging.info(f"{round(time.time() - start_time)} seconds")
                        else:
                            logging.info("No files to merge and upload")
                    logging.info(f"Now merge all bands together for {current_file}")

                    file_band_pattern = Path(current_file).stem + "_band"
                    logging.info(
                        f"Path looking for -> {merged_path}/**/*_{file_band_pattern}*.tif"
                    )
                    all_bands_files = glob(
                        f"{merged_path}/**/*_{file_band_pattern}*.tif", recursive=True
                    )
                    vrt_options = gdal.BuildVRTOptions(
                        resampleAlg="average", separate=True
                    )
                    logging.info(f"Bands to merge in order {sorted(all_bands_files)}")
                    vrt_raster = gdal.BuildVRT(
                        "/vsimem/composed.vrt",
                        sorted(all_bands_files),
                        options=vrt_options,
                    )
                    raster_sharpened_path = os.path.join(
                        work_path, "raster_sharpened_merge"
                    )
                    os.makedirs(raster_sharpened_path, exist_ok=True)
                    raster_merged_vrt_path = os.path.join(
                        raster_sharpened_path, Path(current_file).stem + "_shp.tif"
                    )
                    gdal.Translate(destName=raster_merged_vrt_path, srcDS=vrt_raster)
                    # UPload Results folder
                    remote_folder_path = str(run_id) + "/"
                    logging.info(
                        f"Destination bucket: {DEFAULT_RUN_BUCKET}, folder reference {raster_merged_vrt_path}"
                    )
                    upload_directory(
                        raster_sharpened_path,
                        remote_folder_path,
                        DEFAULT_RUN_BUCKET,
                    )
                    logging.info("Done")


if __name__ == "__main__":
    __process_random_folder()

