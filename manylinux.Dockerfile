FROM quay.io/pypa/manylinux2010_x86_64

COPY manylinux-build.sh .
RUN mkdir -p dist

ENTRYPOINT [ "./manylinux-build.sh" ]
