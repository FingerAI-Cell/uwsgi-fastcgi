FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# uWSGI 설정
ENV UWSGI_PROCESSES=2
ENV UWSGI_THREADS=4
ENV UWSGI_MASTER=1
ENV UWSGI_SOCKET=/tmp/stats.sock
ENV UWSGI_CHMOD_SOCKET=666
ENV UWSGI_VACUUM=1
ENV UWSGI_DIE_ON_TERM=1
ENV UWSGI_WSGI_FILE=app.py
ENV UWSGI_CALLABLE=app

CMD ["uwsgi", "--ini", "/app/uwsgi.ini"] 