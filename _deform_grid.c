#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL Deform_ARRAY_API
#include <numpy/arrayobject.h>
#include "from_scipy.h"
#include "deform.h"

/* Converts a Python array-like object into a behaved input array. */
static int
NI_ObjectToInputArray(PyObject *object, PyArrayObject **array)
{
      int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
          *array = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags,
                                                                 NULL);
              return *array != NULL;
}

/* Converts a Python array-like object into a behaved output array. */
static int
NI_ObjectToOutputArray(PyObject *object, PyArrayObject **array)
{
    #ifdef HAVE_WRITEBACKIFCOPY
        int flags = NPY_ARRAY_BEHAVED_NS | NPY_ARRAY_WRITEBACKIFCOPY;
    #else
        int flags = NPY_ARRAY_BEHAVED_NS | NPY_ARRAY_UPDATEIFCOPY;
    #endif
    /*
     * This would also be caught by the PyArray_CheckFromAny call, but
     * we check it explicitly here to provide a saner error message.
     */
    if (PyArray_Check(object) &&
            !PyArray_ISWRITEABLE((PyArrayObject *)object)) {
        PyErr_SetString(PyExc_ValueError, "output array is read-only.");
    return 0;
    }
    /*
     * If the input array is not aligned or is byteswapped, this call
     * will create a new aligned, native byte order array, and copy the
     * contents of object into it. For an output array, the copy is
     * unnecessary, so this could be optimized. It is very easy to not
     * do NPY_ARRAY_UPDATEIFCOPY right, so we let NumPy do it for us
     * and pay the performance price.
     */
    *array = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags,
                                                   NULL);
    return *array != NULL;
}

static PyObject *Py_DeformGrid(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL;
    PyArrayObject *displacement = NULL;
    int mode, order;
    double cval;

    if (!PyArg_ParseTuple(args, "O&O&O&iid",
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToInputArray, &displacement,
                          NI_ObjectToOutputArray, &output,
                          &order, &mode, &cval))
        goto exit;

    DeformGrid(input, displacement, output, order, (NI_ExtendMode)mode, cval);
    #ifdef HAVE_WRITEBACKIFCOPY
        PyArray_ResolveWritebackIfCopy(output);
    #endif

exit:
    Py_XDECREF(input);
    Py_XDECREF(output);
    Py_XDECREF(displacement);
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyMethodDef module_methods[] = {
    {"deform_grid", (PyCFunction)Py_DeformGrid, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_deform_grid",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__deform_grid(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}

