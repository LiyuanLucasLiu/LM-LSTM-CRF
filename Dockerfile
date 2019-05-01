From te_server:latest

RUN apk add -u python3 && \
	python3 -m ensurepip && \
	apk add -u python3-dev

RUN apk add -u libc6-compat
RUN apk add --virtual .build-deps gcc musl-dev g++
RUN pip3 install cython

COPY requirements.txt /

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN python3 -c "import platform; print(platform.python_version())"
RUN pip3 install -r /requirements.txt

WORKDIR /root
COPY * ./
COPY model model


CMD /bin/sh
