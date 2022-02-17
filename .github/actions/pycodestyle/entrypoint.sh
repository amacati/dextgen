#!/bin/sh

pycodestyle /github/workspace
rc=$?
if [ $rc -ne 0 ] ; then
    echo "PyCodeStyle check failed"
    exit 1
fi

exit $rc