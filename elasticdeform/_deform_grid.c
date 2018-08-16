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

/* Like NI_ObjectToInputArray, but with special handling for Py_None. */
static int
NI_ObjectToOptionalInputArray(PyObject *object, PyArrayObject **array)
{
      if (object == Py_None) {
                *array = NULL;
                        return 1;
                            }
          return NI_ObjectToInputArray(object, array);
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
    PyObject *inputList = NULL, *outputList = NULL;
    PyArrayObject **inputs = NULL, **outputs = NULL;
    PyArrayObject *displacement = NULL, *output_offset = NULL;
    PyArrayObject *orders = NULL, *modes = NULL, *cvals = NULL;
    int *order = NULL, *mode = NULL;
    double *cval = NULL;
    int i;
    Py_ssize_t ninputs = 0;

    if (!PyArg_ParseTuple(args, "O!O&O&O!O&O&O&",
                          &PyList_Type, &inputList,
                          NI_ObjectToInputArray, &displacement,
                          NI_ObjectToOptionalInputArray, &output_offset,
                          &PyList_Type, &outputList,
                          NI_ObjectToInputArray, &orders,
                          NI_ObjectToInputArray, &modes,
                          NI_ObjectToInputArray, &cvals))
        goto exit;

    ninputs = PyList_Size(inputList);
    // TODO check size of outputlist
    // TODO check malloc
    // TODO check size of inputs, outputs
    inputs = malloc(ninputs * sizeof(PyArrayObject*));
    outputs = malloc(ninputs * sizeof(PyArrayObject*));
    for(i = 0; i < ninputs; i++) {
        if (!NI_ObjectToInputArray(PyList_GetItem(inputList, i), &inputs[i]))
            goto exit;
        Py_INCREF(inputs[i]);
        if (!NI_ObjectToOutputArray(PyList_GetItem(outputList, i), &outputs[i]))
            goto exit;
        Py_INCREF(outputs[i]);
    }

    // TODO check size and type of orders, modes, cvals
    // TODO check malloc
    order = malloc(ninputs * sizeof(int));
    mode = malloc(ninputs * sizeof(int));
    cval = malloc(ninputs * sizeof(double));
    for(i = 0; i < ninputs; i++) {
        order[i] = *(int *)PyArray_GETPTR1(orders, i);
        mode[i] = *(int *)PyArray_GETPTR1(modes, i);
        cval[i] = *(double *)PyArray_GETPTR1(cvals, i);
    }

    // TODO check output_offset type (should be npy_intp)

    // TODO
    DeformGrid(ninputs, inputs, displacement, output_offset,
               outputs, order, mode, cval);
    #ifdef HAVE_WRITEBACKIFCOPY
        for(i = 0; i < ninputs; i++) {
            PyArray_ResolveWritebackIfCopy(outputs[i]);
        }
    #endif

exit:
    Py_XDECREF(displacement);
    Py_XDECREF(output_offset);
    Py_XDECREF(orders);
    Py_XDECREF(modes);
    Py_XDECREF(cvals);
    for(i = 0; i < ninputs; i++) {
        if (inputs)
            Py_XDECREF(inputs[i]);
        if (outputs)
            Py_XDECREF(outputs[i]);
    }
    free(inputs);
    free(outputs);
    free(order);
    free(mode);
    free(cval);
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

