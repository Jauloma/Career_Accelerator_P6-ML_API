FROM python:3.8
WORKDIR /Sepsis_App
COPY ./requirements.txt ./
RUN pip install --timeout=1200 -r requirements.txt
COPY ./src/Sepsis_App/ .


EXPOSE 7680
CMD ["python", "main.py"]