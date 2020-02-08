#!/usr/bin/env bash

PACKAGE=copulae
WORKDIR=/${PACKAGE}

mkdir -p ${WORKDIR} && cd ${WORKDIR}

# Compile wheels
for PY_VER in "37" "38"; do

  if [[ $PY_VER == "38" ]]; then
    INNER_VER=$PY_VER
  else
    INNER_VER=${PY_VER}m
  fi

  "/opt/python/cp${PY_VER}-cp${INNER_VER}/bin/pip" install numpy cython scipy
  "/opt/python/cp${PY_VER}-cp${INNER_VER}/bin/pip" wheel ${WORKDIR} -w /wheelhouse
done

mkdir -p ${WORKDIR}/dist

# Bundle external shared libraries into the wheels
for whl in /wheelhouse/${PACKAGE}-*.whl; do
  auditwheel repair "$whl" --plat manylinux1_x86_64 -w ${WORKDIR}/dist/
done
