FROM ubuntu:16.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
	wget \
	curl \
	python-dev \
	build-essential libssl-dev libffi-dev \
	libxml2-dev libxslt1-dev zlib1g-dev \
	nginx \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

RUN pip --no-cache-dir install \
	numpy \
	scipy \
	sklearn \
	pandas \
	flask \
	gevent \
	gunicorn \
	pystan

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY folder_all_data /opt/program
WORKDIR /opt/program
