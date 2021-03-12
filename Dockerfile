FROM tensorflow/tensorflow:latest-gpu

WORKDIR /ml_project/

RUN apt update && apt install -y wget && \
    wget --progress=bar:force:noscroll -nc http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz -O food-101.tar.gz && \
    tar xvzf food-101.tar.gz && \
    rm food-101.tar.gz && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install \
    numpy \
    keras \
    jupyterlab  \
    opencv-python \
    sklearn \
    matplotlib \
    pandas

RUN git clone https://github.com/mhd-medfa/Food-101-Classification.git

CMD jupyter-lab Food-101-Classification/ --allow-root --ip 0.0.0.0 --port 8080
