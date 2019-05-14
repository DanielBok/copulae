FROM quay.io/pypa/manylinux2010_x86_64

COPY manylinux-build.sh .

ENTRYPOINT [ "bash", "./manylinux-build.sh" ]
