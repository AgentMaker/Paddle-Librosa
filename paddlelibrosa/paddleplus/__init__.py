import paddle
import paddle.nn.initializer

def fold(inputs, output_size, kernel_size, stride, padding = 0):
    """Combines an array of sliding local blocks into a large containing tensor
    """

    B, D, L = inputs.shape
    H, W = output_size[0], output_size[1]
    C = int(D / (kernel_size[0] * kernel_size[1]))
    out_h = (H + 2*padding - kernel_size[0]) // stride[0] + 1
    out_w = (W + 2*padding - kernel_size[1]) // stride[1] + 1

    inputs = inputs.reshape([B, C, kernel_size[0], kernel_size[1], out_h, out_w])

    img = paddle.zeros([B, C, H + 2 * padding + stride[0] - 1,
                        W + 2 * padding + stride[1] - 1], dtype=inputs.dtype)

    for y in range(kernel_size[0]):
        y_max = y + stride[0] * out_h
        for x in range(kernel_size[1]):
            x_max = x + stride[1] * out_w
            img[:, :, y:y_max:stride[0], x:x_max:stride[1]] += inputs[:, :, y, x, :, :]

    return img[:, :, padding: H + padding, padding: W + padding]