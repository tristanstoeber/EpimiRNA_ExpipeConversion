FROM jupyter/tensorflow-notebook:latest

USER root

RUN apt-get update

USER jovyan

# Downgrade numpy to a version compatible with the other packages
RUN pip install "numpy<2"

# Install the other packages
RUN pip install odfpy exdir expipe pyedflib mat73

