import numpy as np

# test_label = np.array([0,1,1,1,0,0,0])
# effort = np.array([1,2,3,2,4,5,1])
# predict_label = [1,1,0,1,0,0,1]#改成<class 'numpy.ndarray'>
# predict_label = np.array(predict_label)


# struct
class Xy:
    def __init__(self):
        self.x = float()
        self.y = float()


class Graph:
    def __init__(self):
        self.pred = Xy()
        self.opt = Xy()
        self.wst = Xy()


graph = Graph()
xy = Xy()


# # effort-aware performance measures
#这里的predict_label应该是缺陷倾向性，即predict_proba 第二列的内容
def rank_measure(predict_label, effort, test_label, test_bug):
    length = len(test_label)

    if 0 in effort:
        # for avoiding effort has zero
        effort = [i + 1 for i in effort]
        #effort = effort + 1

    effort = np.array(effort)
    test_label = np.array(test_label)
    test_bug = np.array(test_bug)

    # CBS+排序
    # print('predict_label:', predict_label)
    print(type(predict_label))
    print(type(predict_label[0]))
    defect_idx = np.where(predict_label > 0.5)  # <class 'numpy.ndarray'>
    # print("defect_idx", defect_idx[0])
    nodefect_idx = np.where(predict_label <= 0.5)

    defect_label = predict_label[defect_idx]  # <class 'numpy.ndarray'>
    # print("defect_label", defect_label)
    nodefect_label = predict_label[nodefect_idx]
    # print("nodefect_label", nodefect_label)

    defect_effort = effort[defect_idx]
    # print("defect_effort", defect_effort)
    nodefect_effort = effort[nodefect_idx]
    # print("nodefect_effort", nodefect_effort)

    preDef = defect_label / defect_effort
    # print("preDef", preDef)
    # print(preDef.shape)
    # print(type(preDef))
    prenoDef = nodefect_label / nodefect_effort
    # print("prenoDef", prenoDef)

    defect_data = np.zeros((len(defect_idx[0]), 4))
    defect_data[:, 0] = preDef.T
    defect_data[:, 1] = defect_effort.T
    defect_data[:, 2] = test_label[defect_idx].T
    # print("test_label", test_label, len(test_label))
    # print("test_bug", test_bug, len(test_bug))
    defect_data[:, 3] = test_bug[defect_idx].T
    # print(defect_data)

    nodefect_data = np.zeros((len(nodefect_idx[0]), 4))
    nodefect_data[:, 0] = prenoDef.T
    nodefect_data[:, 1] = nodefect_effort.T
    nodefect_data[:, 2] = test_label[nodefect_idx].T
    nodefect_data[:, 3] = test_bug[nodefect_idx].T
    # print(nodefect_data)

    # sorted函数可以根据多个key值进行数据的排序，按照key值传入的顺序依次进行，当前面元素相同时，根据后面的key值继续排序。
    defect_data = sorted(defect_data, key=lambda x: (-x[0]))  # 根据第一个值降序排序，若第一个值相等则根据第二个值升序排序
    defect_data = np.array(defect_data)
    # print("defect_data", defect_data.shape)
    nodefect_data = sorted(nodefect_data, key=lambda x: (-x[0], x[1]))  # 根据第一个值降序排序，若第一个值相等则根据第二个值升序排序
    nodefect_data = np.array(nodefect_data)
    # print("nodefect_data", nodefect_data.shape)

    data = np.zeros((len(test_label), 4))
    if len(defect_idx[0]) == 0:
        data = nodefect_data
    elif len(nodefect_idx[0]) == 0:
        data = defect_data
    else:
        data[0:len(defect_idx[0]), :] = defect_data
        # print(data[0:len(defect_idx[0]), :].shape)
        data[len(defect_idx[0]):len(test_label), :] = nodefect_data

    # print("length", length)
    # print("data.shape",data.shape)

    cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPofb = computeMeasure(data, length)

    # computing popt
    pred, graph.pred = computeArea(data, length)

    # actual defect density
    actual_density = test_label / effort

    # optimal model
    data = np.zeros(shape=(len(test_label), 3))
    data[:, 0] = actual_density
    data[:, 1] = effort
    data[:, 2] = test_label
    data = sorted(data, key=lambda x: (-x[0], x[1]))
    opt, graph.opt = computeArea(data, length)

    # worst model
    data = np.zeros(shape=(len(test_label), 3))
    data[:, 0] = actual_density
    data[:, 1] = effort
    data[:, 2] = test_label
    data = sorted(data, key=lambda x: (x[0], -x[1]))
    wst, graph.wst = computeArea(data, length)

    if opt - wst != 0:
        Popt = (pred - wst) / (opt - wst)
    else:
        Popt = 0.5

    # print("Popt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA", Popt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA)

    return Popt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPofb
    # return cErecall, cIFA


def computeMeasure(data, length):
    cumXs = np.cumsum(data[:, 1])
    cumYs = np.cumsum(data[:, 2])
    cumBugs = np.cumsum(data[:, 3])
    Xs = cumXs / cumXs[length - 1]

    idx = next(iter(np.where(Xs >= 0.2)[0]), -1)
    # pos=idx
    pos = idx + 1

    Erecall = cumYs[idx] / cumYs[length - 1]
    Epofb = cumBugs[idx] / cumBugs[length - 1]

    Eprecision = cumYs[idx] / pos

    if Erecall + Eprecision != 0:
        Efmeasure = 2 * Erecall * Eprecision / (Erecall + Eprecision)
    else:
        Efmeasure = 0

    PMI = pos / length

    Iidx = next(iter(np.where(cumYs == 1)[0]), -1)
    IFA = Iidx + 1

    return Erecall, Eprecision, Efmeasure, PMI, IFA, Epofb


def computeArea(data, length):
    data = np.array(data)
    cumXs = np.cumsum(data[:, 1])
    cumYs = np.cumsum(data[:, 2])

    Xs = cumXs / cumXs[length - 1]
    Ys = cumYs / cumYs[length - 1]

    xy.x = Xs
    xy.y = Ys
    area = np.trapz(xy.x, xy.y)

    return area, xy
