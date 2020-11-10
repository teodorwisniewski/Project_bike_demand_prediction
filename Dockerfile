FROM python:3.8-rc-alpine

COPY app.py /app/app.py
COPY scripts /app/scripts
COPY resources /app/resources
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN apk update
RUN echo "http://dl-8.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
RUN apk --no-cache --update-cache add gcc gfortran build-base wget freetype-dev libpng-dev openblas-dev
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/app/app.py]