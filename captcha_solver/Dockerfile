FROM python:3.9.14-bullseye

# Install dependencies for librosa and OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
 && apt-get install -y --no-install-recommends python3-opencv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


# copy rnb.pem /etc/ssl/certs/
# copy rnb.cer /usr/local/share/ca-certificates/
# run update-ca-certificates

copy requirements /requirements

WORKDIR /requirements

# RUN apt-get update && apt-get install -y --no-install-recommends python3-opencv ; exit 0
# RUN apt-get install -y --no-install-recommends python3-opencv ; exit 0
# RUN apt-get install -y --no-install-recommends python3-opencv ; exit 0
# RUN apt-get install -y --no-install-recommends python3-opencv

RUN pip install --upgrade pip

# flask
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  --no-cache-dir --no-binary=:all: --require-hashes -r requirements_flask.txt

# librosa
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  librosa==0.9.2

# COPY rnb.pem /rnb.pem
# ENV PIP_CERT=/rnb.pem

# -- install Tensorflow in virtualenv because of conflict versions of numpy library
RUN mkdir /tensorflow && python3 -m venv /tensorflow
RUN . /tensorflow/bin/activate \
  && pip install tensorflow-cpu==2.11.0 --trusted-host pypi.org --trusted-host files.pythonhosted.org

ENV PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"

COPY app /app

WORKDIR /app

ENTRYPOINT ["flask", "--app", "main", "run", "--host", "0.0.0.0", "--port", "80"]
