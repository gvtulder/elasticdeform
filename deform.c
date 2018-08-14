#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL Deform_ARRAY_API
#include <numpy/arrayobject.h>
#include "from_scipy.h"
#include "deform.h"


/* map a coordinate outside the borders, according to the requested
     boundary condition: */
static double
map_coordinate(double in, npy_intp len, int mode)
{
    if (in < 0) {
        switch (mode) {
        case NI_EXTEND_MIRROR:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz2 = 2 * len - 2;
                in = sz2 * (npy_intp)(-in / sz2) + in;
                in = in <= 1 - len ? in + sz2 : -in;
            }
            break;
        case NI_EXTEND_REFLECT:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz2 = 2 * len;
                if (in < -sz2)
                    in = sz2 * (npy_intp)(-in / sz2) + in;
                in = in < -len ? in + sz2 : -in - 1;
            }
            break;
        case NI_EXTEND_WRAP:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz = len - 1;
                // Integer division of -in/sz gives (-in mod sz)
                // Note that 'in' is negative
                in += sz * ((npy_intp)(-in / sz) + 1);
            }
            break;
        case NI_EXTEND_NEAREST:
            in = 0;
            break;
        case NI_EXTEND_CONSTANT:
            in = -1;
            break;
        }
    } else if (in > len-1) {
        switch (mode) {
        case NI_EXTEND_MIRROR:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz2 = 2 * len - 2;
                in -= sz2 * (npy_intp)(in / sz2);
                if (in >= len)
                    in = sz2 - in;
            }
            break;
        case NI_EXTEND_REFLECT:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz2 = 2 * len;
                in -= sz2 * (npy_intp)(in / sz2);
                if (in >= len)
                    in = sz2 - in - 1;
            }
            break;
        case NI_EXTEND_WRAP:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz = len - 1;
                in -= sz * (npy_intp)(in / sz);
            }
            break;
        case NI_EXTEND_NEAREST:
            in = len - 1;
            break;
        case NI_EXTEND_CONSTANT:
            in = -1;
            break;
        }
    }

    return in;
}

int
get_spline_interpolation_weights(double x, int order, double *weights)
{
    int i;
    double y, z, t;

    /* Convert x to the delta to the middle knot. */
    x -= floor(order & 1 ? x : x + 0.5);
    y = x;
    z = 1.0 - x;

    switch (order) {
        case 1:
            /* 0 <= x < 1*/
            weights[0] = 1.0 - x;
            break;
        case 2:
            /* -0.5 < x <= 0.5 */
            weights[1] = 0.75 - x * x;
            /* For weights[0] we'd normally do:
             *
             *   y = 1 + x  # 0.5 < y <= 1.5
             *   yy = 1.5 - y  # yy = 0.5 - x
             *   weights[0] = 0.5 * yy * yy
             *
             * So we set y = 0.5 - x directly instead.
             */
            y = 0.5 - x;
            weights[0] = 0.5 * y * y;
            break;
        case 3:
            /* y = x, 0 <= y < 1 */
            weights[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
            /* z = 1 - x, 0 < z <= 1 */
            weights[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
            /*
             * For weights[0] we would normally do:
             *
             *   y += 1.0  # y = 1.0 + x, 1 <= y < 2
             *   yy = 2.0 - y  # yy = 1 - x
             *   weights[0] = yy * yy * yy / 6.0
             *
             * But we already have yy in z.
             */
            weights[0] = z * z * z / 6.0;
            break;
        case 4:
            /* -0.5 < x <= 0.5 */
            t = x * x;
            weights[2] = t * (t * 0.25 - 0.625) + 115.0 / 192.0;
            /* y = 1 + x, 0.5 < y <= 1.5 */
            y = 1.0 + x;
            weights[1] = y * (y * (y * (5.0 - y) / 6.0 - 1.25) + 5.0 / 24.0) +
                         55.0 / 96.0;
            /* z = 1 - x, 0.5 <= z < 1.5 */
            weights[3] = z * (z * (z * (5.0 - z) / 6.0 - 1.25) + 5.0 / 24.0) +
                         55.0 / 96.0;
            /*
             * For weights[0] we would normally do:
             *
             *   y += 1.0  # y = 2.0 + x, 1.5 <= y < 2.5
             *   yy = 2.5 - y  # yy = 0.5 - x
             *  weights[0] = yy**4 / 24.0
             *
             * So we set y = 0.5 - x directly instead.
             */
            y = 0.5 - x;
            t = y * y;
            weights[0] = t * t / 24.0;
            break;
        case 5:
            /* y = x, 0 <= y < 1 */
            t = y * y;
            weights[2] = t * (t * (0.25 - y / 12.0) - 0.5) + 0.55;
            /* z = 1 - x, 0 < z <= 1 */
            t = z * z;
            weights[3] = t * (t * (0.25 - z / 12.0) - 0.5) + 0.55;
            /* y = 1 + x, 1 <= y < 2 */
            y += 1.0;
            weights[1] = y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) - 1.75)
                              + 0.625) + 0.425;
            /* z = 2 - x, 1 < z <= 2 */
            z += 1.0;
            weights[4] = z * (z * (z * (z * (z / 24.0 - 0.375) + 1.25) - 1.75)
                              + 0.625) + 0.425;
            /* For weights[0] we would normally do:
             *
             *   y += 1.0  # y = 2.0 + x, 2 <= y < 3
             *   yy = 3.0 - y  # yy = 1.0 - x
             *   weights[0] = yy**5 / 120.0
             *
             * So we set y = 2.0 - y = 1.0 - x directly instead.
             */
            y = 1.0 - x;
            t = y * y;
            weights[0] = y * t * t / 120.0;
            break;
        default:
            return 1; /* Unsupported spline order. */
    }

    /* All interpolation weights add to 1.0, so use it for the last one. */
    weights[order] = 1.0;
    for (i = 0; i < order; ++i) {
        weights[order] -= weights[i];
    }

    return 0;
}

/* copy row of coordinate array from location at _p to _coor */
#define CASE_MAP_COORDINATES(_TYPE, _type, _p, _coor, _rank, _stride) \
case _TYPE:                                                           \
{                                                                     \
    npy_intp _hh;                                                     \
    for (_hh = 0; _hh < _rank; ++_hh) {                               \
        _coor[_hh] = *(_type *)_p;                                    \
        _p += _stride;                                                \
    }                                                                 \
}                                                                     \
break

#define CASE_INTERP_COEFF(_TYPE, _type, _coeff, _pi, _idx) \
case _TYPE:                                                \
    _coeff = *(_type *)(_pi + _idx);                       \
    break

#define CASE_INTERP_OUT(_TYPE, _type, _po, _t) \
case _TYPE:                                    \
    *(_type *)_po = (_type)_t;                 \
    break

#define CASE_INTERP_OUT_UINT(_TYPE, _type, _po, _t)  \
case NPY_##_TYPE:                                    \
    _t = _t > 0 ? _t + 0.5 : 0;                      \
    _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t; \
    _t = _t < 0 ? 0 : t;                             \
    *(_type *)_po = (_type)_t;                       \
    break

#define CASE_INTERP_OUT_INT(_TYPE, _type, _po, _t)   \
case NPY_##_TYPE:                                    \
    _t = _t > 0 ? _t + 0.5 : _t - 0.5;               \
    _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t; \
    _t = _t < NPY_MIN_##_TYPE ? NPY_MIN_##_TYPE : t; \
    *(_type *)_po = (_type)_t;                       \
    break

int DeformGrid(PyArrayObject* input, PyArrayObject* displacement, PyArrayObject* output_offset,
               PyArrayObject* output, int order, int mode, double cval)
{
    char *po, *pi, *pd = NULL;
    npy_intp **edge_offsets = NULL, **data_offsets = NULL, filter_size, dfilter_size;
    npy_intp **dedge_offsets = NULL, **ddata_offsets = NULL;
    npy_intp ftmp[NPY_MAXDIMS], *fcoordinates = NULL, *foffsets = NULL;
    npy_intp dftmp[NPY_MAXDIMS], *dfcoordinates = NULL, *dfoffsets = NULL;
    npy_intp kk, hh, ll, jj;
    npy_intp size;
    double **splvals = NULL, **dsplvals = NULL;
    npy_intp idimensions[NPY_MAXDIMS], istrides[NPY_MAXDIMS];
    npy_intp odimensions[NPY_MAXDIMS];
    npy_intp ddimensions[NPY_MAXDIMS], dstrides[NPY_MAXDIMS];
    npy_intp ooffsets[NPY_MAXDIMS];
    npy_intp ncontrolpoints[NPY_MAXDIMS];
    NI_Iterator io;
    int irank = 0;

    /* spline order for deplacement */
    int dorder = 3;

    NPY_BEGIN_THREADS_DEF;

    NPY_BEGIN_THREADS;

    for(kk = 0; kk < PyArray_NDIM(input); kk++) {
        idimensions[kk] = PyArray_DIM(input, kk);
        istrides[kk] = PyArray_STRIDE(input, kk);
        odimensions[kk] = PyArray_DIM(output, kk);
    }
    irank = PyArray_NDIM(input);

    for (kk = 0; kk < PyArray_NDIM(displacement); kk++) {
        ddimensions[kk] = PyArray_DIM(displacement, kk);
        dstrides[kk] = PyArray_STRIDE(displacement, kk);
    }

    /* check if the output is cropped */
    for(kk = 0; kk < PyArray_NDIM(input); kk++) {
        ooffsets[kk] = 0;
    }
    if (output_offset) {
        for(kk = 0; kk < PyArray_NDIM(input); kk++) {
            ooffsets[kk] = *(npy_intp *)(PyArray_GETPTR1(output_offset, kk));
        }
    }

    /* number of control points in each dimension */
    for (kk = 0; kk < irank; kk++) {
        ncontrolpoints[kk] = PyArray_DIM(displacement, kk + 1);
    }

    /* offsets used at the borders: */
    edge_offsets = malloc(irank * sizeof(npy_intp*));
    data_offsets = malloc(irank * sizeof(npy_intp*));
    if (NPY_UNLIKELY(!edge_offsets || !data_offsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank; jj++)
        data_offsets[jj] = NULL;
    for(jj = 0; jj < irank; jj++) {
        data_offsets[jj] = malloc((order + 1) * sizeof(npy_intp));
        if (NPY_UNLIKELY(!data_offsets[jj])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }

    /* offsets in desplacement used at the borders: */
    dedge_offsets = malloc(irank * sizeof(npy_intp*));
    ddata_offsets = malloc(irank * sizeof(npy_intp*));
    if (NPY_UNLIKELY(!dedge_offsets || !ddata_offsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank; jj++)
        ddata_offsets[jj] = NULL;
    for(jj = 0; jj < irank; jj++) {
        ddata_offsets[jj] = malloc((dorder + 1) * sizeof(npy_intp));
        if (NPY_UNLIKELY(!ddata_offsets[jj])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }

    npy_int perimeter = 0;
    for(jj = 0; jj < irank; jj++) {
        perimeter += odimensions[jj];
    }
    /* will hold the deplacement spline coefficients: */
    dsplvals = malloc(irank * perimeter * sizeof(double*));
    if (NPY_UNLIKELY(!dsplvals)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank * perimeter; jj++)
        dsplvals[jj] = NULL;
    for(jj = 0; jj < irank * perimeter; jj++) {
        dsplvals[jj] = malloc((dorder + 1) * sizeof(double));
        if (NPY_UNLIKELY(!dsplvals[jj])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }

    /* will hold the spline coefficients: */
    splvals = malloc(irank * sizeof(double*));
    if (NPY_UNLIKELY(!splvals)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank; jj++)
        splvals[jj] = NULL;
    for(jj = 0; jj < irank; jj++) {
        splvals[jj] = malloc((order + 1) * sizeof(double));
        if (NPY_UNLIKELY(!splvals[jj])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }

    filter_size = 1;
    for(jj = 0; jj < irank; jj++)
        filter_size *= order + 1;

    dfilter_size = 1;
    for(jj = 0; jj < irank; jj++)
        dfilter_size *= dorder + 1;

    /* initialize output iterator: */
    if (!NI_InitPointIterator(output, &io))
        goto exit;

    /* get data pointers: */
    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    pd = (void *)PyArray_DATA(displacement);

    /* make a table of all possible coordinates within the spline filter: */
    fcoordinates = malloc(irank * filter_size * sizeof(npy_intp));
    /* make a table of all offsets within the spline filter: */
    foffsets = malloc(filter_size * sizeof(npy_intp));
    if (NPY_UNLIKELY(!fcoordinates || !foffsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank; jj++)
        ftmp[jj] = 0;
    kk = 0;
    for(hh = 0; hh < filter_size; hh++) {
        for(jj = 0; jj < irank; jj++)
            fcoordinates[jj + hh * irank] = ftmp[jj];
        foffsets[hh] = kk;
        for(jj = irank - 1; jj >= 0; jj--) {
            if (ftmp[jj] < order) {
                ftmp[jj]++;
                kk += istrides[jj];
                break;
            } else {
                ftmp[jj] = 0;
                kk -= istrides[jj] * order;
            }
        }
    }

    /* make a table of all possible coordinates within the spline filter: */
    dfcoordinates = malloc(irank * dfilter_size * sizeof(npy_intp));
    /* make a table of all offsets within the spline filter: */
    dfoffsets = malloc(dfilter_size * sizeof(npy_intp));
    if (NPY_UNLIKELY(!dfcoordinates || !dfoffsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank; jj++)
        dftmp[jj] = 0;
    kk = 0;
    for(hh = 0; hh < dfilter_size; hh++) {
        for(jj = 0; jj < irank; jj++)
            dfcoordinates[jj + hh * irank] = dftmp[jj];
        dfoffsets[hh] = kk;
        for(jj = irank - 1; jj >= 0; jj--) {
            if (dftmp[jj] < dorder) {
                dftmp[jj]++;
                kk += dstrides[jj + 1];
                break;
            } else {
                dftmp[jj] = 0;
                kk -= dstrides[jj + 1] * dorder;
            }
        }
    }

    /* compute control point */
    kk = 0;
    for(hh = 0; hh < irank; hh++) {
        for(jj = 0; jj < odimensions[hh]; jj++) {
            double cp = (double)(ncontrolpoints[hh] - 1) * (double)(jj + ooffsets[hh]) / (double)(idimensions[hh] - 1);
            get_spline_interpolation_weights(cp, dorder, dsplvals[kk]);
            kk++;
        }
    }

    size = PyArray_SIZE(output);
    for(kk = 0; kk < size; kk++) {
        double t = 0.0;
        int constant = 0, edge = 0;
        npy_intp offset = 0;

        /* compute deplacement on this dimension */
        double dd = 0.0;
        int dedge = 0;
        npy_intp ddoffset = 0;
        for(jj = 0; jj < irank; jj++) {
            /* assumption: by definition, cp is inside the deplacement array */
            double cp = (double)(ncontrolpoints[jj] - 1) * (double)(io.coordinates[jj] + ooffsets[jj]) / (double)(idimensions[jj] - 1);
            /* find the filter location along this axis: */
            npy_intp start;
            if (dorder & 1) {
                start = (npy_intp)floor(cp) - dorder / 2;
            } else {
                start = (npy_intp)floor(cp + 0.5) - dorder / 2;
            }
            /* get the offset to the start of the filter: */
            ddoffset += dstrides[jj + 1] * start;
            if (start < 0 || start + dorder >= ddimensions[jj + 1]) {
                /* implement border mapping, if outside border: */
                dedge = 1;
                dedge_offsets[jj] = ddata_offsets[jj];
                for(ll = 0; ll <= dorder; ll++) {
                    npy_intp idx = start + ll;
                    npy_intp len = ddimensions[jj + 1];
                    if (len <= 1) {
                        idx = 0;
                    } else {
                        npy_intp s2 = 2 * len - 2;
                        if (idx < 0) {
                            idx = s2 * (int)(-idx / s2) + idx;
                            idx = idx <= 1 - len ? idx + s2 : -idx;
                        } else if (idx >= len) {
                            idx -= s2 * (int)(idx / s2);
                            if (idx >= len)
                                idx = s2 - idx;
                        }
                    }
                    /* calculate and store the offests at this edge: */
                    dedge_offsets[jj][ll] = dstrides[jj + 1] * (idx - start);
                }
            } else {
                /* we are not at the border, use precalculated offsets: */
                dedge_offsets[jj] = NULL;
            }
        }

        /* iterate over axes: */
        for(hh = 0; hh < irank; hh++) {
            /* compute displacement */
            npy_intp *dff = dfcoordinates;
            const int type_num = PyArray_TYPE(displacement);
            dd = 0.0;
            for(jj = 0; jj < dfilter_size; jj++) {
                double coeff = 0.0;
                npy_intp idx = 0;

                if (NPY_UNLIKELY(dedge)) {
                    for(ll = 0; ll < irank; ll++) {
                        if (dedge_offsets[ll])
                            idx += dedge_offsets[ll][dff[ll]];
                        else
                            idx += dff[ll] * dstrides[ll + 1];
                    }
                } else {
                    idx = dfoffsets[jj];
                }
                idx += dstrides[0] * hh;
                idx += ddoffset;
                switch (type_num) {
                    CASE_INTERP_COEFF(NPY_BOOL, npy_bool,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_UBYTE, npy_ubyte,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_USHORT, npy_ushort,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_UINT, npy_uint,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_ULONG, npy_ulong,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_ULONGLONG, npy_ulonglong,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_BYTE, npy_byte,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_SHORT, npy_short,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_INT, npy_int,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_LONG, npy_long,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_LONGLONG, npy_longlong,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_FLOAT, npy_float,
                                      coeff, pd, idx);
                    CASE_INTERP_COEFF(NPY_DOUBLE, npy_double,
                                      coeff, pd, idx);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError,
                                    "data type not supported");
                    goto exit;
                }
                /* calculate the interpolated value: */
                double **cur_dsplvals = dsplvals;
                for(ll = 0; ll < irank; ll++) {
                    if (dorder > 0)
                        coeff *= cur_dsplvals[io.coordinates[ll]][dff[ll]];
                    cur_dsplvals += odimensions[ll];
                }
                dd += coeff;
                dff += irank;
            }


            /* compute the coordinate: coordinate of output voxel io.coordinates[hh] + displacement dd */
            /* if the input coordinate is outside the borders, map it: */
            double cc = map_coordinate(io.coordinates[hh] + ooffsets[hh] + dd, idimensions[hh], mode);
            if (cc > -1.0) {
                /* find the filter location along this axis: */
                npy_intp start;
                if (order & 1) {
                    start = (npy_intp)floor(cc) - order / 2;
                } else {
                    start = (npy_intp)floor(cc + 0.5) - order / 2;
                }
                /* get the offset to the start of the filter: */
                offset += istrides[hh] * start;
                if (start < 0 || start + order >= idimensions[hh]) {
                    /* implement border mapping, if outside border: */
                    edge = 1;
                    edge_offsets[hh] = data_offsets[hh];
                    for(ll = 0; ll <= order; ll++) {
                        npy_intp idx = start + ll;
                        npy_intp len = idimensions[hh];
                        if (len <= 1) {
                            idx = 0;
                        } else {
                            npy_intp s2 = 2 * len - 2;
                            if (idx < 0) {
                                idx = s2 * (int)(-idx / s2) + idx;
                                idx = idx <= 1 - len ? idx + s2 : -idx;
                            } else if (idx >= len) {
                                idx -= s2 * (int)(idx / s2);
                                if (idx >= len)
                                    idx = s2 - idx;
                            }
                        }
                        /* calculate and store the offests at this edge: */
                        edge_offsets[hh][ll] = istrides[hh] * (idx - start);
                    }
                } else {
                    /* we are not at the border, use precalculated offsets: */
                    edge_offsets[hh] = NULL;
                }
                get_spline_interpolation_weights(cc, order, splvals[hh]);
            } else {
                /* we use the constant border condition: */
                constant = 1;
                break;
            }
        }

        if (!constant) {
            npy_intp *ff = fcoordinates;
            const int type_num = PyArray_TYPE(input);
            t = 0.0;
            for(hh = 0; hh < filter_size; hh++) {
                double coeff = 0.0;
                npy_intp idx = 0;

                if (NPY_UNLIKELY(edge)) {
                    for(ll = 0; ll < irank; ll++) {
                        if (edge_offsets[ll])
                            idx += edge_offsets[ll][ff[ll]];
                        else
                            idx += ff[ll] * istrides[ll];
                    }
                } else {
                    idx = foffsets[hh];
                }
                idx += offset;
                switch (type_num) {
                    CASE_INTERP_COEFF(NPY_BOOL, npy_bool,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_UBYTE, npy_ubyte,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_USHORT, npy_ushort,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_UINT, npy_uint,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_ULONG, npy_ulong,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_ULONGLONG, npy_ulonglong,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_BYTE, npy_byte,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_SHORT, npy_short,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_INT, npy_int,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_LONG, npy_long,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_LONGLONG, npy_longlong,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_FLOAT, npy_float,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_DOUBLE, npy_double,
                                      coeff, pi, idx);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError,
                                    "data type not supported");
                    goto exit;
                }
                /* calculate the interpolated value: */
                for(ll = 0; ll < irank; ll++)
                    if (order > 0)
                        coeff *= splvals[ll][ff[ll]];
                t += coeff;
                ff += irank;
            }
        } else {
            t = cval;
        }
        /* store output value: */
        switch (PyArray_TYPE(output)) {
            CASE_INTERP_OUT(NPY_BOOL, npy_bool, po, t);
            CASE_INTERP_OUT_UINT(UBYTE, npy_ubyte, po, t);
            CASE_INTERP_OUT_UINT(USHORT, npy_ushort, po, t);
            CASE_INTERP_OUT_UINT(UINT, npy_uint, po, t);
            CASE_INTERP_OUT_UINT(ULONG, npy_ulong, po, t);
            CASE_INTERP_OUT_UINT(ULONGLONG, npy_ulonglong, po, t);
            CASE_INTERP_OUT_INT(BYTE, npy_byte, po, t);
            CASE_INTERP_OUT_INT(SHORT, npy_short, po, t);
            CASE_INTERP_OUT_INT(INT, npy_int, po, t);
            CASE_INTERP_OUT_INT(LONG, npy_long, po, t);
            CASE_INTERP_OUT_INT(LONGLONG, npy_longlong, po, t);
            CASE_INTERP_OUT(NPY_FLOAT, npy_float, po, t);
            CASE_INTERP_OUT(NPY_DOUBLE, npy_double, po, t);
        default:
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
        }
        NI_ITERATOR_NEXT(io, po);
    }

 exit:
    NPY_END_THREADS;
    free(edge_offsets);
    free(dedge_offsets);
    if (data_offsets) {
        for(jj = 0; jj < irank; jj++)
            free(data_offsets[jj]);
        free(data_offsets);
    }
    if (ddata_offsets) {
        for(jj = 0; jj < irank; jj++)
            free(ddata_offsets[jj]);
        free(ddata_offsets);
    }
    if (dsplvals) {
        for(jj = 0; jj < irank * perimeter; jj++)
            free(dsplvals[jj]);
        free(dsplvals);
    }
    if (splvals) {
        for(jj = 0; jj < irank; jj++)
            free(splvals[jj]);
        free(splvals);
    }
    free(foffsets);
    free(fcoordinates);
    free(dfoffsets);
    free(dfcoordinates);
    return PyErr_Occurred() ? 0 : 1;
}
