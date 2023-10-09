FROM python:3.11.4-bookworm

RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.26.0
RUN pip3 install pandas==2.1.1
RUN pip3 install ipykernel
RUN pip3 install seaborn==0.12.2
RUN pip3 install scikit-learn==1.3.1
RUN pip3 install ppscore
RUN pip3 install matplotlib==3.7.3
RUN pip3 install mlflow
RUN pip3 install pytest
# RUN pip3 install dash_bootstrap_components
# RUN pip3 install shap
# RUN pip3 install dash

# open when push to DockerHub
# WORKDIR /root/source_code/app

# Open when push to DockerHub
COPY ./source_code /root/source_code
# Close when push to DockerHub and Open when using in local
CMD tail -f /dev/null

