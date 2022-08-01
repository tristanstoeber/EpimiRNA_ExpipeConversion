FROM jupyter/tensorflow-notebook:latest

USER root

RUN apt-get update

USER jovyan
RUN pip install odfpy
RUN pip install exdir
RUN pip install expipe
RUN pip install pyedflib
RUN pip install mat73

