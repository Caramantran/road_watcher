FROM continuumio/miniconda3

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    python3-opencv \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Africa/Casablanca

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY environment.yml .

RUN conda env create -f environment.yml \
    && conda clean -afy

RUN conda run -n yolo_env pip install ultralytics hikvisionapi python-dotenv

COPY . .

RUN chmod -R 777 /usr/src/app

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "yolo_env", "python", "main.py"]
