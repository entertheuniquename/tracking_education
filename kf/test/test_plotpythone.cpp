#include <Python.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional.hpp>

#include <gtest/gtest.h>

class PrintTree {

    std::string PathToFile;
    std::string ModuleName;
    std::string FuncName;

    PyObject* py_module = nullptr;
    PyObject* py_func = nullptr;

    bool connectToFunc(const std::string& funcName) {
        if (FuncName==funcName && py_func) {
            return true;
        }
        FuncName = funcName;
        if (!py_module) {
            std::cerr << "Failed to load " << ModuleName << " module" << std::endl;
            return false;
        }
        py_func = PyObject_GetAttrString(py_module, funcName.c_str());
        if (!py_func || !PyCallable_Check(py_func)) {
            std::cerr << "Failed to load " << FuncName << " function" << std::endl;
            return false;
        }
        return true;
    }

public:

    PrintTree() {
    }

    ~PrintTree() {
        if (py_func) {
            Py_XDECREF(py_func);
        }
        if (py_module) {
            Py_DECREF(py_module);
        }
        Py_Finalize();
    }

    bool ConnectToModule(const std::string& pathToFile,
                         const std::string& moduleName) {
        PathToFile = pathToFile;
        ModuleName = moduleName;

        Py_Initialize();
        PyObject* sys_module = PyImport_ImportModule("sys");
        if (sys_module) {
            PyObject* sys_path = PyObject_GetAttrString(sys_module, "path");
            if (sys_path && PyList_Check(sys_path)) {
                PyObject* path = PyUnicode_FromString(pathToFile.c_str());
                PyList_Append(sys_path, path);
                Py_DECREF(path);
            } else {
                std::cerr << "Failed to get sys.path" << std::endl;
            }
            Py_XDECREF(sys_path);
            Py_DECREF(sys_module);
        } else {
            std::cerr << "Failed to import sys module" << std::endl;
            return false;
        }

        PyObject* sys_path = PyObject_GetAttrString(sys_module, "path");
        std::cout << "Contents of sys.path:" << std::endl;
        Py_ssize_t size = PyList_Size(sys_path);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* path_item = PyList_GetItem(sys_path, i);
            if (path_item && PyUnicode_Check(path_item)) {
                const char* path_str = PyUnicode_AsUTF8(path_item);
                std::cout << path_str << std::endl;
            }
        }

        py_module = PyImport_ImportModule(moduleName.c_str());
        if (!py_module) {
            PyErr_Print();
            std::cerr << "Failed to load " << moduleName << " module" << std::endl;
            return false;
        }

        return true;
    }


    bool Plot(const std::string& funcName,
              const std::map<std::string, int>& data) {
        if (!connectToFunc(funcName)) {
            return false;
        }
        PyObject* py_dict = PyDict_New();
        for (const auto& pair : data) {
            PyObject* key = PyUnicode_FromString(pair.first.c_str());
            PyObject* value = PyLong_FromLong(pair.second);
            PyDict_SetItem(py_dict, key, value);
            Py_DECREF(key);
            Py_DECREF(value);
        }
        if (py_func && PyCallable_Check(py_func)) {
            PyObject* py_result = PyObject_CallFunctionObjArgs(py_func, py_dict, NULL);
            if (!py_result) {
                PyErr_Print();
                std::cerr << "Failed to call function" << std::endl;
                return false;
            }
            Py_XDECREF(py_result);
        }
        Py_DECREF(py_dict);
        return true;
    }

    struct Node {
        struct Connect {
            int Id;
            double Distance;
        };
        int Id;
        int TrackID;
        double Time;
        double Score;
        double ScoreMax;
        int DetID;
        Connect Parent;
        std::vector<Connect> Children;
    };

    PyObject* ConvertNodesToPyObject(const std::vector<Node>& nodes) {
        // Создание пустого списка Python
        PyObject* py_list = PyList_New(0);
        if (!py_list) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
            return nullptr;
        }

        // Проход по каждому элементу вектора Node
        for (const auto& node : nodes) {
            // Создание словаря Python для текущего узла
            PyObject* py_node_dict = PyDict_New();
            if (!py_node_dict) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to create Python dictionary");
                Py_DECREF(py_list);
                return nullptr;
            }

            // Заполнение словаря атрибутами узла
            PyDict_SetItemString(py_node_dict, "Id", PyLong_FromLong(node.Id));
            PyDict_SetItemString(py_node_dict, "TrackID", PyLong_FromLong(node.TrackID));
            PyDict_SetItemString(py_node_dict, "Time", PyFloat_FromDouble(node.Time));
            PyDict_SetItemString(py_node_dict, "Score", PyFloat_FromDouble(node.Score));
            PyDict_SetItemString(py_node_dict, "ScoreMax", PyFloat_FromDouble(node.ScoreMax));
            PyDict_SetItemString(py_node_dict, "DetID", PyLong_FromLong(node.DetID));

            // Создание словаря Python для родительского узла
            PyObject* py_parent_dict = PyDict_New();
            if (!py_parent_dict) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to create Python dictionary");
                Py_DECREF(py_list);
                Py_DECREF(py_node_dict);
                return nullptr;
            }
            PyDict_SetItemString(py_parent_dict, "Id", PyLong_FromLong(node.Parent.Id));
            PyDict_SetItemString(py_parent_dict, "Distance", PyFloat_FromDouble(node.Parent.Distance));
            PyDict_SetItemString(py_node_dict, "Parent", py_parent_dict);
            Py_DECREF(py_parent_dict);

            // Создание списка Python для детей узла
            PyObject* py_children_list = PyList_New(node.Children.size());
            if (!py_children_list) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to create Python list");
                Py_DECREF(py_list);
                Py_DECREF(py_node_dict);
                return nullptr;
            }
            // Заполнение списка детей
            for (size_t i = 0; i < node.Children.size(); ++i) {
                PyObject* py_child_dict = PyDict_New();
                if (!py_child_dict) {
                    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python dictionary");
                    Py_DECREF(py_list);
                    Py_DECREF(py_node_dict);
                    Py_DECREF(py_children_list);
                    return nullptr;
                }
                PyDict_SetItemString(py_child_dict, "Id", PyLong_FromLong(node.Children[i].Id));
                PyDict_SetItemString(py_child_dict, "Distance", PyFloat_FromDouble(node.Children[i].Distance));
                PyList_SetItem(py_children_list, i, py_child_dict);
            }
            PyDict_SetItemString(py_node_dict, "Children", py_children_list);
            Py_DECREF(py_children_list);

            // Добавление словаря узла в список
            PyList_Append(py_list, py_node_dict);
            Py_DECREF(py_node_dict);
        }

        return py_list;
    }

    bool Plot(const std::string& funcName,
              PyObject* var1, PyObject* var2) {
        if (!connectToFunc(funcName)) {
            return false;
        }
        if (py_func && PyCallable_Check(py_func)) {
            PyObject* py_result = PyObject_CallFunctionObjArgs(py_func, var1, var2, NULL);
            if (!py_result) {
                PyErr_Print();
                std::cerr << "Failed to call function" << std::endl;
                return false;
            }
            Py_XDECREF(py_result);
        }
        return true;
    }
};

TEST (PlotPython, Node) {
    std::string pathToFile {/*ROOT_DIR*/"/home/ivan/projects/kf"};
    pathToFile += "/scripts";

    PrintTree pt;
    pt.ConnectToModule(pathToFile, "tree_vizual");
/*
    int Id;
    int TrackID;
    double Time;
    double Score;
    double ScoreMax;
    int DetID;
*/
    std::vector<PrintTree::Node> nodes = {
        {1, 10, 1.5, 0.9,  7., 10, {}, {{2, 2.3}}},
        {2, 20, 3.0, 0.8, 7.5, -1, {1, 2.3}, {}}
    };

    auto* nodesPyObj1 = pt.ConvertNodesToPyObject(nodes);
    auto* nodesPyObj2 = pt.ConvertNodesToPyObject(nodes);

    //pt.Plot("PlotGraph", nodesPyObj1, nodesPyObj2);
    //pt.Plot("PlotGraph", nodesPyObj1, nodesPyObj2);


    Py_DECREF(nodesPyObj1);
    Py_DECREF(nodesPyObj2);
}



