FROM --platform=amd64 intel/oneapi

RUN apt-get update && \
    apt-get install -y python3-dev ninja-build python3-full python3-pip && \
    apt-get clean

WORKDIR /opt/Shamrock_src

ADD repo.tar.gz .

RUN ls -la

RUN ./env/new-env --machine docker.intel_oneapi --builddir build_intel_docker -- && \
    cd build_intel_docker && \
    pip3 install -v --break-system-packages .

# The base image ships some Python packages through apt (e.g. typing_extensions)
# that pip cannot uninstall (no RECORD file) when a dependency needs a newer
# version. --ignore-installed installs pip's copies in /usr/local instead,
# which shadow the apt ones without touching them.
RUN pip3 install -v --break-system-packages --ignore-installed jupyter matplotlib numpy scipy
EXPOSE 8888

WORKDIR /work
