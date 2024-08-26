FROM python:3.12.5-slim-bullseye
WORKDIR /usr/src/app/
RUN apt update && apt upgrade -y
RUN python -m pip install -U pip
RUN pip install flask_restx flask_cors pymysql pandas scikit_learn
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY ./api.py /usr/src/app/api.py
CMD ["python", "api.py"]