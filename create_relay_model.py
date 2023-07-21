import numpy as np
import tvm
from tvm import relay

def create_relay_model():
   inp0 = relay.var("input", shape=[1, 224, 224, 3], dtype="float32")
    weight_datas = np.random.random([3, 3, 3, 32])
    out = relay.nn.conv2d(
        inp0,
        relay.const(weight_datas, dtype="float32"),
        strides=[2, 2],
        channels=32,
        kernel_size=(3, 3),
        padding=[0, 0, 1, 1],
        data_layout="NHWC",
        kernel_layout="HWIO",
    )

    dtype = "float32"
    beta = relay.const(np.random.random([32]), dtype=dtype)
    gamma = relay.const(np.random.random([32]), dtype=dtype)
    moving_mean = relay.const(np.random.random([32]), dtype=dtype)
    moving_var = relay.const(np.random.random([32]), dtype=dtype)
    out = relay.nn.batch_norm(out, gamma, beta, moving_mean, moving_var, axis=3, epsilon=0.001)
    #out = list(out)
    #out = out[0] if len(out) == 1 else relay.Tuple(out)
    out = relay.clip(out[0], a_min=0, a_max=6)


    weight_datas2 = np.random.random([3, 3, 32, 1])
    out = relay.nn.conv2d(
        out,
        relay.const(weight_datas2, dtype="float32"),
        channels=32,
        kernel_size=[3, 3],
        padding=[1, 1, 1, 1],
        groups = 32,
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    beta = relay.const(np.random.random([32]), dtype=dtype)
    gamma = relay.const(np.random.random([32]), dtype=dtype)
    moving_mean = relay.const(np.random.random([32]), dtype=dtype)
    moving_var = relay.const(np.random.random([32]), dtype=dtype)
    out = relay.nn.batch_norm(out, gamma, beta, moving_mean, moving_var, axis=3, epsilon=0.001)
    ##out = list(out)
    #out = out[0] if len(out) == 1 else relay.Tuple(out)
    out = relay.clip(out[0], a_min=0, a_max=6)

    #out = list(out)
    #out = out[0] if len(out) == 1 else relay.Tuple(out)
    func = relay.Function([inp0], out)
    mod = tvm.IRModule.from_expr(func)
    with open("./relay_tiny.rly", "w") as f:
        f.write(mod.astext(True))

if __name__ == "__main__":
  create_relay_model()
