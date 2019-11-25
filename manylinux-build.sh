#!/usr/bin/env bash

PACKAGE=copulae
WORKDIR=/${PACKAGE}

mkdir -p ${WORKDIR} && cd ${WORKDIR}

# Compile wheels
for PY_VER in "36" "37" "38"; do
  "/opt/python/cp${PY_VER}-cp${PY_VER}m/bin/pip" install numpy cython scipy
  "/opt/python/cp${PY_VER}-cp${PY_VER}m/bin/pip" wheel ${WORKDIR} -w /wheelhouse
done

mkdir -p ${WORKDIR}/dist

# Bundle external shared libraries into the wheels
for whl in /wheelhouse/${PACKAGE}-*.whl; do
  auditwheel repair "$whl" --plat manylinux1_x86_64 -w ${WORKDIR}/dist/
done
