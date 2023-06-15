#import tensorflow as tf
import onnxruntime as ort
import onnx
import numpy as np
from onnx import helper
import sys
#import torch

# vis: https://netron.app/
# model file: onnx/backend/test/data/node/test_hardmax_axis_1/model.onnx
def main(model_path, in_paths, in_names):
    # input0=np.random.random([3,4,5])*10
    # input0 = input0.astype(np.uint16)
    # input0.tofile("in0.bin")
    """
    model_path = 'xxx.onnx'
    #inp1 = np.random.random([5,4,3,78]).astype(np.float32)*10#(np.random.ranf((10)).astype(np.float32)) * 5
    #inp1=np.array([5,6,7,8,9]).astype(np.float32)
    #inp2=np.array([1,2,3]).astype(np.uint16)
    """

    # load model
    model = onnx.load(model_path)

    # check model
    onnx.checker.check_model(model)

    # create session
    ort_sess = ort.InferenceSession(model_path)

    out_names = [o.name for o in ort_sess.get_outputs()]
    ins = ort_sess.get_inputs()
    assert(len(ins) == len(in_names))
    assert(len(in_paths) == len(in_names))

    # prepare input
    ins_dict = dict()
    for in_idx in range(len(ins)):
        name = ins[in_idx].name
        data = np.load(in_paths[in_idx], allow_pickle=True)
        print(f"in[{in_idx}], {name}, {data.shape}")
        ins_dict[name] = data

    # run
    outs_data = ort_sess.run(out_names, ins_dict)

    # get output
    for out_idx in range(len(outs_data)):
        out = outs_data[out_idx]
        print(f"outs_data[{out_idx}] with shape {out.shape}")
    return


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"./{sys.argv[0]} <onnx-model-path> <in0_path,in1_path,in2_path> <in0_name,in1_name,in2_name>")
        exit(1)

    # model
    model_path = sys.argv[1]

    # input paths
    in_paths_str = sys.argv[2]
    in_paths = in_paths_str.split(",")

    # input names
    in_names_str = sys.argv[2]
    in_names = in_names_str.split(",")

    # check
    assert(len(in_names) == len(in_paths))

    main(model_path, in_paths, in_names)
