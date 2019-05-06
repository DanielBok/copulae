#!/usr/bin/env bash

WORKDIR=/copulae
PLAT=${PLAT:-manylinux2010_x86_64}

git clone https://github.com/DanielBok/copulae.git ${WORKDIR}
cd ${WORKDIR}
git checkout $(git describe --tags)

# Compile wheels
for PY_VER in "36" "37"; do
    "/opt/python/cp${PY_VER}-cp${PY_VER}m/bin/pip" install numpy cython
    "/opt/python/cp${PY_VER}-cp${PY_VER}m/bin/pip" wheel ${WORKDIR} -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat ${PLAT} -w wheelhouse/
done

find wheelhouse/ -name "copulae-*-manylinux2010_x86_64.whl" -exec cp -t /dist {} +;
