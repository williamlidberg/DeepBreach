FROM tensorflow/tensorflow:2.16.1-gpu
RUN apt-get update

# setup GDAL
RUN apt-get install libgdal-dev -y
RUN pip install GDAL==3.3.2

RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install tifffile
RUN pip install geopandas
RUN pip install imagecodecs
RUN pip install whitebox-workflows
RUN pip install whitebox
RUN pip install imageio
RUN pip install splitraster
RUN pip install seaborn
RUN pip install scikit-learn
RUN pip install rasterio
RUN pip install jupyterlab
RUN pip install visualkeras

# create mount points for data and source code in container's start directory
RUN mkdir /workspace
RUN mkdir /workspace/data
RUN mkdir /workspace/code
RUN mkdir /workspace/temp
RUN mkdir /workspace/temp_inference

WORKDIR /workspace/code

