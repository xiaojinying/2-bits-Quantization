import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


def expand_group_params(input_tensor, group_size, scale_param, zero_param):
    """
    将 group-wise 的 scale / zero 按照 group_size 展开，并 reshape 成与输入维度兼容的形状
    """
    # groupsize=-1 时表示一个 group 覆盖整列
    group_size = input_tensor.shape[1] if group_size == -1 else group_size

    expanded_scale = None
    expanded_zero = None

    if scale_param is not None:
        # 在列维度重复，每组有 group_size 个元素
        expanded_scale = torch.repeat_interleave(scale_param, repeats=group_size, dim=1)
        # 这里把 scale reshape 成 [C, D, 1, 1, ...] 方便后续广播到高维张量
        broadcast_shape = [expanded_scale.shape[0], expanded_scale.shape[1]] + [1] * (len(input_tensor.shape) - 2)
        expanded_scale = expanded_scale.reshape(broadcast_shape)

    if zero_param is not None:
        expanded_zero = torch.repeat_interleave(zero_param, repeats=group_size, dim=1)
        broadcast_shape = [expanded_zero.shape[0], expanded_zero.shape[1]] + [1] * (len(input_tensor.shape) - 2)
        expanded_zero = expanded_zero.reshape(broadcast_shape)

    return expanded_scale, expanded_zero


def compute_channel_error(x_float, x_quant, hessian):
    """
    计算每个通道的量化误差:
        err_i = (q_i - x_i)^T H (q_i - x_i)
    其中 H 是 Hessian 近似矩阵
    """
    assert x_float.ndim == 2
    assert x_quant.ndim == 2
    assert hessian.ndim == 2
    # 直接用矩阵乘法，比逐元素三重循环要快很多
    diff = x_quant - x_float
    err = torch.matmul(torch.matmul(diff, hessian), diff.t()).diag()
    return err  # [num_channel]


def round_half_away_from_zero(x):
    """
    自定义 round：
    - 对于 .5 情况，行为为“远离 0”而不是默认的“偶数舍入”
    这样可以保证不同实现之间的行为更一致
    """
    y = torch.round(x)
    idx = (y - x).abs() == 0.5
    if idx.any():
        x_clone = x.clone()
        x_clone[idx] += 0.5
        return torch.round(x_clone)
    else:
        return y


class SimpleAdamOptimizer(object):
    """
    极简版 Adam 优化器，仅用于本文件内部的坐标/权重优化
    """
    def __init__(self, param_tensor: torch.Tensor, lr: float) -> None:
        self.param = param_tensor
        self.lr = lr
        self.t = 0  # 迭代步数
        # 一阶与二阶动量，形状与参数一致
        self.exp_avg = torch.zeros_like(param_tensor, memory_format=torch.preserve_format)
        self.exp_avg_sq = torch.zeros_like(param_tensor, memory_format=torch.preserve_format)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1.0e-5

    def step(self, grad):
        """
        执行一次 Adam 更新：param = param - lr * Adam(grad)
        """
        self.t += 1
        # 一阶动量：指数滑动平均
        self.exp_avg.lerp_(grad, 1 - self.beta1)
        # 二阶动量：平方梯度的指数滑动平均
        self.exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        denom = (self.exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
        self.param.addcdiv_(self.exp_avg, denom, value=-step_size)


@torch.no_grad()
def coordinate_descent_with_box(x, hessian_matrix, left_bound, right_bound, start_index: int = 0):
    """使用坐标下降（coordinate descent）在 box 约束 [left, right] 下最小化 x^T H x
    其中 H 是对称正定矩阵。
    该版本按维度顺序更新，如果 H 的对角线已按降序排序，收敛会更好。
    x: [num_channel, dim]
    hessian_matrix: [dim, dim]
    left_bound: [num_channel, dim]
    right_bound: [num_channel, dim]
    """
    # 初始梯度 g = xH
    grad = torch.matmul(x, hessian_matrix)  # [num_channel, dim]
    for i in range(start_index, x.shape[1]):
        diag_val = hessian_matrix[i, i]
        # 去掉当前维的贡献，得到“其他维度”的梯度
        other_grad = grad - torch.outer(x[:, i], hessian_matrix[i, :])
        # 一维二次项最优解：x_i = -g_i / H_ii，并投影到 [left, right]
        x[:, i] = (-other_grad[:, i] / diag_val).clamp(min=left_bound[:, i], max=right_bound[:, i])
        # 更新新的整体梯度
        grad = other_grad + torch.outer(x[:, i], hessian_matrix[i, :])
    return x


@torch.no_grad()
def optimize_int_weight_v3(scale_out, zero_out, hessian_matrix, sym, min_bound, max_bound,
                           x0, x_init=None, max_iter=0, lr=1.e-3,
                           max_inner_iter=0, round_fn="gptq"):
    """
    主入口：给定 scale_out / zero_out 和 Hessian，优化 int 权重（含连续优化 + 离散 rounding）
    """
    lower = min_bound * scale_out + zero_out
    upper = max_bound * scale_out + zero_out
    left = torch.minimum(lower, upper)
    right = torch.maximum(lower, upper)

    # 初始连续解 w
    w = x0.clone() if x_init is None else x_init.clone()
    optimizer = SimpleAdamOptimizer(w, lr)

    # 如果是 GPTQ 模式，则跳过连续 Adam 迭代，直接做 rounding
    max_iter = 0 if round_fn == "gptq" else max_iter
    for _ in range(max_iter):
        # 直接手工计算梯度 (w - x0) H，比前向 + 反向快很多
        optimizer.step(torch.matmul(w - x0, hessian_matrix))
        w.data.clamp_(min=left, max=right)

    assert round_fn in ("gptq", "train")
    round_func = rounding_with_gptq if round_fn == "gptq" else rounding_with_training

    # 进行最终的 rounding（GPTQ 风格或训练式）
    f1, w = round_func(
        w,
        scale_out,
        zero_out,
        sym,
        min_bound,
        max_bound,
        x0=x0,
        H00=hessian_matrix,
        left=left,
        right=right,
        max_iter=max_inner_iter,
        opt=optimizer,
    )

    # 使用 Hessian 计算量化误差
    loss = compute_channel_error(f1, x0, hessian_matrix)
    # 将最终浮点解 f1 映射到整数区间
    w_int = round_half_away_from_zero(
        safe_divide_elementwise(f1 - zero_out, scale_out)
    ).clamp(min=min_bound, max=max_bound)
    return w_int, w, loss


def rounding_with_gptq(weight_float, scale_out, zero_out, sym, min_bound, max_bound, H00=None, **kwargs):
    """
    GPTQ 风格的逐列 rounding：
    - 使用近似的 H^-1，将 rounding 产生的误差向后续列传播，减少总体误差
    """
    new_weight = weight_float.clone()
    Q = weight_float.clone()

    mean_diag = torch.mean(torch.diag(H00)).double()
    diag = torch.arange(weight_float.shape[1], device=weight_float.device)

    # 给对角线增加不同强度的 damp，寻找一个数值稳定的 H
    for t in [0.01, 0.1, 1.0, 10.0, 100.0]:
        H = H00.clone().double()
        damp = t * mean_diag
        H[diag, diag] += damp
        # 先求逆，再做 Cholesky 分解（参考 GPTQ 论文里的技巧）
        H = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H = torch.linalg.cholesky(H, upper=True)
        if not H.isnan().any():
            break
    if H.isnan().any():
        from IPython import embed
        embed(header="nan appears!")
    H = H.to(weight_float.dtype)
    Hinv = H

    for i in range(weight_float.shape[1]):
        w_col = weight_float[:, i]
        d = Hinv[i, i]

        new_weight[:, i] = w_col

        q_int = round_half_away_from_zero(
            safe_divide_elementwise(w_col - zero_out[:, i], scale_out[:, i])
        ).clamp(min=min_bound, max=max_bound)
        q_val = q_int * scale_out[:, i] + zero_out[:, i]
        q_val = q_val.flatten()
        Q[:, i] = q_val

        # 误差在后续列上传播，减小整体损失
        err1 = (w_col - q_val) / d
        weight_float[:, i:] -= err1.unsqueeze(1).matmul(Hinv[i, i:].unsqueeze(0))
    return Q, new_weight


def rounding_with_training(w, scale_out, zero_out, sym, min_bound, max_bound, H00=None, x0=None,
                           left=None, right=None, lr=0.001, max_iter=100, opt=None, **kwargs):
    """
    训练式 rounding：
    - 固定前面已经量化的列
    - 对后面的列用 Adam 做若干步连续优化，再做 rounding
    """
    exp_avg, exp_avg_sq = opt.exp_avg, opt.exp_avg_sq
    beta1, beta2 = opt.beta1, opt.beta2
    eps = 1.0e-4
    step = opt.t
    new_weight = w.clone()

    for i in range(0, w.shape[1] - 1):
        new_weight[:, i] = w[:, i]
        q_int = round_half_away_from_zero(
            safe_divide_elementwise(w[:, i] - zero_out[:, i], scale_out[:, i])
        ).clamp(min=min_bound, max=max_bound)
        q_val = q_int * scale_out[:, i] + zero_out[:, i]
        w[:, i] = q_val.flatten()

        # 截断 exp_avg, exp_avg_sq，保持形状与剩余列一致
        exp_avg, exp_avg_sq = exp_avg[:, 1:], exp_avg_sq[:, 1:]
        l, r = left[:, (i + 1):], right[:, (i + 1):]
        h_sub = H00[:, (i + 1):]

        for _ in range(max_iter):
            step += 1
            grad = torch.matmul(w, h_sub) if x0 is None else torch.matmul(w - x0, h_sub)
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            w[:, (i + 1):].addcdiv_(exp_avg, denom, value=-step_size)
            w[:, (i + 1):].clamp_(min=l, max=r)

    new_weight[:, -1] = w[:, -1]
    # 最终对所有列做一次整数化并反量化
    w = round_half_away_from_zero(
        safe_divide_elementwise(w - zero_out, scale_out)
    ).clamp(min=min_bound, max=max_bound) * scale_out + zero_out
    return w, new_weight


def safe_divide_elementwise(x, y: torch.Tensor):
    """
    安全除法：按元素计算 x / y，避免除以 0
    """
    sign = torch.sign(y)
    sign[sign == 0] = 1
    # 避免 0 除，假设为 float32 精度
    y_safe = y.abs().clamp(min=1.0e-15) * sign
    return x / y_safe


def fake_quantize(x, scale, zero, min_bound, max_bound,
                  scale_out=None, zero_out=None, return_int=False, zero_no_shift=False):
    """
    通用量化函数：
    - 如果 return_int=True，返回整数 q
    - 否则返回反量化值
    """
    if zero_no_shift:
        q_int = torch.clamp(round_half_away_from_zero(x / scale) + zero, min=min_bound, max=max_bound)
    else:
        q_int = torch.clamp(
            round_half_away_from_zero((x - zero) / scale),
            min=min_bound,
            max=max_bound,
        )

    if return_int:
        return q_int

    if scale_out is not None:
        return q_int * scale_out + zero_out

    if zero_no_shift:
        return scale * (q_int - zero)

    return scale * q_int + zero


def compute_error_with_scale_and_zero(x, scale, zero, scale_out, zero_out, sym,
                                      min_bound, max_bound, groupsize, h=None, x0=None,
                                      zero_no_shift=False):
    """
    给定 scale / zero / scale_out / zero_out，评估对应的量化误差
    """
    scale_expanded, zero_expanded = expand_group_params(x, groupsize, scale, zero)
    scale_out_expanded, zero_out_expanded = expand_group_params(x, groupsize, scale_out, zero_out)

    x_ref = x.clone() if x0 is None else x0
    q_val = fake_quantize(
        x,
        scale_expanded,
        zero_expanded,
        min_bound=min_bound,
        max_bound=max_bound,
        scale_out=scale_out_expanded,
        zero_out=zero_out_expanded,
        zero_no_shift=zero_no_shift,
    )
    err = compute_channel_error(x_ref, q_val, h)
    return err


def solve_two_linear_equations(a, b, c, d, e, f):
    """
    求解 2×2 线性方程组:
        a x + b y = c
        d x + e y = f
    返回 x, y
    """
    deno = a * e - b * d
    sign = torch.sign(deno)
    sign[sign == 0] = 1.0
    bu = sign * 1e-5
    singular = deno.abs() < 1e-5
    deno[singular] = bu[singular]
    x = (c * e - b * f) / deno
    y = (a * f - c * d) / deno
    return x, y


class WeightQuantizer(nn.Module):
    """
    权重量化器：提供搜索 scale / zero 的多种方法
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def configure(self, bits, perchannel=False, sym=True, grid=200,
                  maxshrink=0.8, thr=0.01, iters_for_scale=4,
                  zero_no_shift=False):
        """
        配置量化器的基本超参数
        """
        # 这里使用有符号量化区间 [-2^{b-1}, 2^{b-1}-1]
        self.max_bound = 2 ** (bits - 1) - 1
        self.min_bound = -self.max_bound - 1

        self.perchannel = perchannel
        self.sym = sym
        self.grid = grid
        self.maxshrink = maxshrink
        self.thr = thr
        self.iters_for_scale = iters_for_scale
        self.zero_no_shift = zero_no_shift

    def find_params(self, x, groupsize=-1, H=None, best=None, x0=None):
        """
        主入口：给定权重 x 和 Hessian H，搜索合适的 scale / zero
        """
        xmin, xmax = self.get_minmax(x, groupsize)
        scale, zero = self.get_scale_zero_by_minmax(xmax, xmin)  # [num_channel, num_group]

        if self.zero_no_shift:
            scale_out, zero_out = scale.clone(), -scale * zero
        else:
            scale_out, zero_out = scale.clone(), zero.clone()

        best = compute_error_with_scale_and_zero(
            x,
            scale,
            zero,
            scale_out,
            zero_out,
            self.sym,
            self.min_bound,
            self.max_bound,
            groupsize,
            H,
            x0=x0,
            zero_no_shift=self.zero_no_shift,
        )

        if self.iters_for_scale == 0:
            return scale, zero, scale_out, zero_out, best

        scale, zero, best, _ = self.get_scale_and_zero(
            xmin,
            xmax,
            x,
            H,
            best=best,
            scale=scale,
            zero=zero,
            groupsize=groupsize,
            x0=x0,
        )

        if self.zero_no_shift:
            scale_out, zero_out = scale.clone(), -scale * zero
        else:
            scale_out, zero_out = scale.clone(), zero.clone()

        if self.iters_for_scale == 1:
            return scale, zero, scale_out, zero_out, best

        try:
            scale_out, zero_out, best = self.get_scale_and_zero_out_group(
                x, scale, zero, H, groupsize, x0=x0
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        return scale, zero, scale_out, zero_out, best

    def get_scale_and_zero_out_group(self, x=None, scale=None, zero=None,
                                     H=None, groupsize=-1, x0=None,
                                     x_int=None, double_precision=True):
        """
        按 group 维度求解每个 group 的最优 scale_out / zero_out
        （解析解，避免逐元素搜索）
        """
        x0 = x if x0 is None else x0
        if groupsize == -1 or groupsize == x0.shape[1]:
            return self.get_scale_and_zero_out(x, scale, zero, H, x0=x0, x_int=x_int)

        if x_int is None:
            x_int = self.get_fake_int_in(x, scale, zero, groupsize=groupsize)

        # 有 group 的情况下，这里默认是不对称量化
        assert not self.sym

        def matmul_three(m1, m2, m3):
            return torch.matmul(torch.matmul(m1, m2.unsqueeze(0)), m3)

        num_channel = x_int.shape[0]
        dim = x_int.shape[1]
        num_group = dim // groupsize
        a = x_int

        # 下面这段构造矩阵 P 的代码比较“丑”，主要是为了节省显存
        torch.cuda.empty_cache()
        A = torch.zeros((num_channel, num_group, dim), dtype=x_int.dtype, device=x_int.device)
        for j in range(num_group):
            A[:, j, j * groupsize: (j + 1) * groupsize] = a[:, j * groupsize: (j + 1) * groupsize]
        AH = torch.matmul(A, H.unsqueeze(0))
        A = torch.transpose(A, 1, 2)
        torch.cuda.empty_cache()
        P11 = torch.matmul(AH, A)
        del A
        torch.cuda.empty_cache()
        I = torch.zeros((num_channel, num_group, dim), dtype=x_int.dtype, device=x_int.device)
        for j in range(num_group):
            I[:, j, j * groupsize: (j + 1) * groupsize] = 1.0
        P12 = torch.matmul(AH, torch.transpose(I, 1, 2))
        del AH
        P22 = matmul_three(I, H, torch.transpose(I, 1, 2))
        del I
        torch.cuda.empty_cache()
        P21 = torch.transpose(P12, 1, 2)
        P = torch.cat([torch.cat((P11, P12), dim=2), torch.cat((P21, P22), dim=2)], dim=1).to(x_int.dtype)
        del P11, P12, P21, P22

        left = torch.matmul(x0, H)  # [channel, dim]
        left = torch.reshape(left, (num_channel, num_group, groupsize))
        up = torch.mul(left, a.reshape(num_channel, num_group, groupsize)).sum(2)  # [channel, num_group]
        down = left.sum(2)  # [channel, num_group]
        y = torch.cat([up, down], dim=1)  # [channel, 2 * num_group]
        P = (P + P.transpose(1, 2)) / 2.0  # 保证 P 是严格对称的

        if double_precision:
            P, y = P.double(), y.double()
        try:
            scale_zero = torch.linalg.solve(P, y)
        except Exception:
            diag = torch.arange(P.shape[-1], device=P.device)
            dP = P[:, diag, diag]  # [num_channel, 2 * num_group]
            damp = 0.01 * torch.mean(dP, dim=1, keepdim=True)  # [num_channel, 1]
            P[:, diag, diag] += damp
            scale_zero = torch.linalg.solve(P, y)
        scale_zero = scale_zero.float()
        scale_out, zero_out = torch.split(scale_zero, num_group, dim=1)

        if self.zero_no_shift:
            zero_out = round_half_away_from_zero(zero_out / scale_out) * scale_out

        scale_out0, zero_out0 = expand_group_params(x_int, groupsize, scale_out, zero_out)
        x_dequant = x_int * scale_out0 if self.sym else x_int * scale_out0 + zero_out0
        err = compute_channel_error(x0, x_dequant, H)
        return scale_out, zero_out, err

    def get_scale_and_zero(self, xmin, xmax, x, h, scale_out=None, zero_out=None,
                           best=None, scale=None, zero=None,
                           groupsize=-1, x0=None):
        """
        在 [xmin, xmax] 的收缩区间上搜索 scale/zero：
        - 通过遍历不同的 shrink p，找到误差最小的那一组参数
        """
        x0 = x if x0 is None else x0
        if best is None:
            best = torch.full([x.shape[0]], float("inf"), device=x.device)
        else:
            best = best.clone()

        if scale is not None:
            scale, zero = scale.clone(), zero.clone()
            eary_exit = False
        else:
            scale, zero = self.get_scale_zero_by_minmax(xmax, xmin)
            eary_exit = True

        bestp = torch.ones(x.shape[0], dtype=xmin.dtype, device=xmin.device)
        for i in range(int(self.maxshrink * self.grid)):  # 例如 0.8 * 200
            p = 1 - i / self.grid
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1, zero1 = self.get_scale_zero_by_minmax(xmax1, xmin1)
            err = compute_error_with_scale_and_zero(
                x,
                scale1,
                zero1,
                scale_out,
                zero_out,
                self.sym,
                self.min_bound,
                self.max_bound,
                groupsize,
                h,
                x0=x0,
                zero_no_shift=self.zero_no_shift,
            )
            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                scale[tmp] = scale1[tmp]
                zero[tmp] = zero1[tmp]
                bestp[tmp] = p
            elif eary_exit:
                return scale, zero, best, bestp
        return scale, zero, best, bestp

    def get_minmax(self, x, groupsize):
        """
        计算每个通道 / 每个 group 内的最小值和最大值
        """
        if self.perchannel:
            x = x.flatten(1)  # [num_channel, dim]
        else:
            x = x.flatten().unsqueeze(0)

        if groupsize == -1 or groupsize == x.shape[1]:
            num_group = 1
            groupsize = x.shape[1]
        else:
            assert x.shape[1] % groupsize == 0
            num_group = x.shape[1] // groupsize

        x = torch.reshape(x, (x.shape[0], num_group, groupsize))
        zero_ref = torch.zeros(x.shape[:2], device=x.device)
        xmin = torch.minimum(x.min(2)[0], zero_ref)  # [num_channel, num_group]
        xmax = torch.maximum(x.max(2)[0], zero_ref)

        if self.sym:
            # 对称量化：上下界取绝对值较大的那个
            xmax = torch.maximum(torch.abs(xmin), xmax)
            neg_mask = xmin < 0
            if torch.any(neg_mask):
                xmin[neg_mask] = -xmax[neg_mask]

        # 避免区间为 [0, 0] 的退化情况
        zero_mask = (xmin == 0) & (xmax == 0)
        xmin[zero_mask] = -1
        xmax[zero_mask] = +1
        return xmin, xmax

    def quant_via_minmax(self, x, groupsize, h, bestp=None):
        """
        使用 minmax + shrink 搜索的一种“快速量化”方式
        """
        xmin, xmax = self.get_minmax(x, groupsize)
        if bestp is None:
            scale, zero, _, bestp = self.get_scale_and_zero(
                xmin, xmax, x, h, groupsize=groupsize
            )
        else:
            shape = [xmax.shape[0]] + [1] * (xmax.dim() - 1)
            bestp1 = torch.reshape(bestp, shape)
            scale, zero = self.get_scale_zero_by_minmax(
                xmax * bestp1, xmin * bestp1
            )  # [num_channel, num_group]

        scale_expanded, zero_expanded = expand_group_params(x, groupsize, scale, zero)
        x_q = fake_quantize(
            x,
            scale_expanded,
            zero_expanded,
            self.min_bound,
            self.max_bound,
            zero_no_shift=self.zero_no_shift,
        )
        return x_q, bestp

    def get_fake_int_in(self, x, scale, zero, groupsize=-1, zero_no_shift=None):
        """
        只做量化到整数，不进行反量化
        """
        scale_expanded, zero_expanded = expand_group_params(x, groupsize, scale, zero)
        if zero_no_shift is None:
            zero_no_shift = self.zero_no_shift
        return fake_quantize(
            x,
            scale_expanded,
            zero_expanded,
            self.min_bound,
            self.max_bound,
            None,
            None,
            return_int=True,
            zero_no_shift=zero_no_shift,
        )

    def get_scale_and_zero_out(self, x, scale, zero, h, x0=None, x_int=None):
        """
        全通道（不分组）情况下，直接解析计算最优的 scale_out / zero_out
        """
        x0 = x if x0 is None else x0
        x_int = self.get_fake_int_in(x, scale, zero) if x_int is None else x_int

        if self.sym:
            scale_out, err = self.get_scale_out_sym(h, x0, x_int, -1)
            # 对称量化时，zero_out 可以视为固定常数（通常是区间中点）
            zero_out = torch.full_like(scale_out, (self.max_bound + self.min_bound + 1) // 2)
            return scale_out, zero_out, err

        # 非对称量化：解析解一个凸二次函数的最优点
        a = x_int  # [out_channel, dim]
        x0 = x0  # [out_channel, dim]
        I = torch.ones_like(a)  # [out_channel, dim]
        h = (h + h.t()) / 2.0  # 保证 h 为对称矩阵 [dim, dim]
        Ih = torch.matmul(I, h)  # [out_channel, dim]
        ah = torch.matmul(a, h)  # [out_channel, dim]

        # 下列变量对应论文中展开后系数矩阵的若干项
        p1 = torch.matmul(ah, a.t()).diag()  # [out_channel]
        p2 = torch.matmul(I, ah.t()).diag()  # [out_channel]
        p3 = torch.matmul(x0, ah.t()).diag()  # [out_channel]
        p4 = p2
        p5 = torch.matmul(Ih, I.t()).diag()
        p6 = torch.matmul(x0, Ih.t()).diag()

        scale_out, zero_out = solve_two_linear_equations(p1, p2, p3, p4, p5, p6)
        if torch.any(torch.isnan(scale_out) | torch.isinf(scale_out)) or torch.any(
            torch.isnan(zero_out) | torch.isinf(zero_out)
        ):
            raise ValueError("Fatal Error")

        scale_out, zero_out = scale_out.unsqueeze(1), zero_out.unsqueeze(1)
        if self.zero_no_shift:
            zero_out = round_half_away_from_zero(zero_out / scale_out) * scale_out

        scale_out0, zero_out0 = expand_group_params(x_int, -1, scale_out, zero_out)
        q = x_int * scale_out0 if self.sym else x_int * scale_out0 + zero_out0
        err = compute_channel_error(x0, q, h)
        return scale_out, zero_out, err

    def get_scale_out_sym(self, h, x0, x_int, groupsize=-1):
        """
        对称量化时的解析 scale_out 计算：
        - 对每个通道，直接用 (x^T H x) / (q^T H q) 的形式求出最优缩放
        """
        assert self.sym
        h = (h + h.t()) / 2.0  # 保证对称 [dim, dim]
        ha = torch.matmul(h, x_int.t())  # [dim, num_channel]
        wha = torch.matmul(x0, ha).diag()  # [num_channel]
        aha = torch.matmul(x_int, ha).diag()  # [num_channel]
        scale_out = safe_divide_elementwise(wha, aha).unsqueeze(1)

        scale_out0, _ = expand_group_params(x_int, groupsize, scale_out, None)
        q = x_int * scale_out0
        err = compute_channel_error(x0, q, h)
        return scale_out, err

    def get_scale_zero_by_minmax(self, xmax, xmin):
        """
        经典 min-max 量化：根据 [xmin, xmax] 区间推导 scale / zero
        """
        scale = (xmax - xmin) / (self.max_bound - self.min_bound)
        if self.sym:
            zero = torch.full_like(scale, (self.max_bound + self.min_bound + 1) // 2)
        elif self.zero_no_shift:
            zero = self.min_bound - round_half_away_from_zero(xmin / scale)
        else:
            zero = xmin - scale * self.min_bound
        return scale, zero

    def free(self):
        """
        预留的清理接口，当前实现中暂不需要释放额外资源
        """
        pass


# 递归收集需要量化的层（Linear / Conv2d / Conv1D）
def collect_quant_layers(module, target_layer_types=nn.Linear, module_name: str = ""):
    """
    递归遍历子模块，收集所有符合类型的层，返回 {层名称: 模块对象} 的字典
    """
    if isinstance(module, target_layer_types):
        return {module_name: module}

    sub_layers = {}
    for child_name, child_module in module.named_children():
        full_child_name = module_name + "." + child_name if module_name != "" else child_name
        sub_layers.update(
            collect_quant_layers(
                child_module,
                target_layer_types=target_layer_types,
                module_name=full_child_name,
            )
        )
    return sub_layers


def move_to_device(data, device):
    """
    通用递归版 .to(device)：
    - 支持 Tensor / dict / list / tuple / nn.Module
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        new_obj = {}
        for k, v in data.items():
            new_obj[k] = move_to_device(v, device)
        return new_obj
    elif isinstance(data, (list, tuple)):
        new_list = [move_to_device(v, device) for v in data]
        if isinstance(data, tuple):
            new_list = tuple(new_list)
        return new_list
    elif isinstance(data, nn.Module):
        return data.to(device)
    return data


def cast_fp16_to_float32(data):
    """
    将对象中的 Tensor 从 fp16/bf16 转为 fp32：
    - 支持 Tensor / dict / list / tuple / nn.Module
    """
    if isinstance(data, torch.Tensor) and data.dtype in (torch.bfloat16, torch.float16):
        return data.float()
    elif isinstance(data, dict):
        new_obj = {}
        for k, v in data.items():
            new_obj[k] = cast_fp16_to_float32(v)
        return new_obj
    elif isinstance(data, (list, tuple)):
        new_list = [cast_fp16_to_float32(v) for v in data]
        if isinstance(data, tuple):
            new_list = tuple(new_list)
        return new_list
    elif isinstance(data, nn.Module):
        # 直接把 module 内部参数 / buffer 的精度提到 fp32
        for p in list(data.parameters()) + list(data.buffers()):
            if p.dtype in (torch.bfloat16, torch.float16):
                p.data = p.data.float()
        return data
    return data


class LayerQuantizerHelper(object):
    """
    用于：
    1. 收集一层的 Hessian 统计信息（通过输入 batch 计算 X^T X）
    2. 基于 GPTQ + 额外优化步骤搜索量化参数
    """

    def __init__(self, target_layer, layer_name: str = ""):
        self.target_layer = target_layer
        weight_matrix = target_layer.weight.data

        # 根据不同类型的层，把权重 reshape 成 [out_features, in_features]
        if isinstance(self.target_layer, nn.Conv2d):
            weight_matrix = weight_matrix.flatten(1)
        elif self.target_layer.__class__.__name__ in ("Conv1D",) and weight_matrix.ndim == 2:
            weight_matrix = weight_matrix.t()
        elif isinstance(self.target_layer, nn.Linear):
            pass
        else:
            raise NotImplementedError("not support yet")

        self.num_rows = weight_matrix.shape[0]
        self.num_cols = weight_matrix.shape[1]

        # 用来累积 Hessian 近似（其实是输入协方差矩阵）
        self.hessian_matrix = 0
        self.num_samples = 0

    def accumulate_batch_stats(self, inputs, outputs, mask):
        """
        用一个 batch 的激活值更新 Hessian 近似：
        H ≈ E[x x^T]
        """
        if inputs.isnan().any():
            print("catch a NAN!!!!!!")
            return

        mask = mask.to(inputs.dtype) if mask is not None else None

        # Linear / Conv1D：统一展平为 [batch, dim] 再转置为 [dim, batch]
        if isinstance(self.target_layer, nn.Linear) or self.target_layer.__class__.__name__ in ("Conv1D",):
            inputs = inputs.reshape((-1, inputs.shape[-1]))  # [batch, dim]
            if mask is not None:
                mask = mask.reshape((-1, 1))  # [batch, 1]
            inputs = inputs * mask if mask is not None else inputs
            inputs = inputs.t()  # [dim, batch]

        # Conv2d：使用 unfold 展开卷积感受野
        elif isinstance(self.target_layer, nn.Conv2d):  # [batch, channel, height, width]
            unfold = nn.Unfold(
                self.target_layer.kernel_size,
                dilation=self.target_layer.dilation,
                padding=self.target_layer.padding,
                stride=self.target_layer.stride,
            )
            inputs = unfold(inputs)  # [B, C*kh*kw, L]
            inputs = inputs.permute(1, 0, 2)  # [C*kh*kw, B, L]
            inputs = inputs.flatten(1)        # [C*kh*kw, B*L]
        else:
            raise NotImplementedError("not support yet")

        # 统计当前 batch 的“有效样本数”
        if mask is not None:
            effective_batch = mask.sum().double()
        else:
            effective_batch = inputs.shape[-1]

        inputs = inputs.double()
        gram_matrix = inputs.matmul(inputs.t())  # X @ X^T

        # 指数滑动平均的方式累积 Hessian
        self.hessian_matrix = (
            self.hessian_matrix * self.num_samples + gram_matrix
        ) / (self.num_samples + effective_batch)
        self.num_samples += effective_batch

    def run_quantization(
        self,
        group_size,
        is_symmetric,
        max_iter_num,
        inner_iters_for_round,
        iters_before_round,
        device,
        learning_rate,
        actorder=True,
        round_fn="gptq",
        perdamp=0.01,
    ):
        """
        主量化过程：
        1. 使用 minmax 初始化 scale/zero
        2. 可选 GPTQ 单轮优化
        3. 多轮 optimize_int_weight_v3 + analytic scale 迭代
        """
        # 拷贝一份权重并搬到指定设备，后续在该副本上做量化搜索
        weight_matrix = self.target_layer.weight.data.clone().detach()
        weight_matrix = weight_matrix.to(device).float()

        # 提示：GPTQ 的样本数最好大于 Hessian 的维度，否则 H 接近奇异
        warn_msg = (
            "The number of samples for GPTQ must be larger than the dimension of Hession matrix;"
            f"Otherwise, H must be a singular matrix and cannot be inverted. nsample {self.num_samples}, columns {self.num_cols}"
        )
        # assert self.num_samples > self.num_cols, warn_msg
        print(warn_msg)

        hessian = self.hessian_matrix.to(device=device, dtype=weight_matrix.dtype)
        del self.hessian_matrix

        # 对角线加上一个小的扰动，防止数值不稳定
        dp = torch.mean(torch.diag(hessian))
        diag_index = torch.arange(hessian.shape[0], device=hessian.device)
        hessian[diag_index, diag_index] += perdamp * dp

        H00 = hessian
        W00 = weight_matrix

        # ===================================================================================================
        # 1) 先用 minmax 找一个初始的 scale/zero，err 作为基准
        iters_for_scale = self.quantizer.iters_for_scale
        self.quantizer.iters_for_scale = 0
        scale, zero, scale_out, zero_out, err = self.quantizer.find_params(
            weight_matrix, groupsize=group_size, H=hessian
        )
        print(f"get scale via minmax, the init loss is {err.mean().item()}")

        # 将 FP 权重量化到 fake int（用于后续迭代）
        w_int = self.quantizer.get_fake_int_in(W00, scale, zero, groupsize=group_size)
        del scale, zero

        # 2) 可选的 GPTQ 单轮优化（max_iter_num >= 1）
        if max_iter_num >= 1:
            # max_iter_num = 1 对应纯 GPTQ
            s0, z0 = expand_group_params(weight_matrix, group_size, scale_out, zero_out)
            h00, w00 = H00.clone(), W00.clone()
            perm, invperm = None, None
            if actorder:
                # 根据 Hessian 对角线大小排序，优先优化重要通道
                perm = torch.argsort(torch.diag(hessian), descending=True)
                invperm = torch.argsort(perm)
                h00 = h00[perm][:, perm]
                s0, z0 = s0[:, perm], z0[:, perm]
                w00 = w00[:, perm]

            w_int0, _, err0 = optimize_int_weight_v3(
                s0,
                z0,
                h00,
                is_symmetric,
                self.quantizer.min_bound,
                self.quantizer.max_bound,
                w00,
                round_fn="gptq",
            )
            if actorder:
                w_int0 = w_int0[:, invperm]

            better_mask = err0 < err
            w_int[better_mask] = w_int0[better_mask]
            err[better_mask] = err0[better_mask]
            success_rate = torch.sum(better_mask) / better_mask.shape[0]
            print(
                f"the success rate of gptq is {success_rate.item()}, the loss is {err.mean().item()}"
            )

        torch.cuda.empty_cache()
        max_inner_iter = inner_iters_for_round
        iter_num = -1

        # 3) 多轮迭代：optimize_int_weight_v3 + analytic_scale 交替
        for iter_num in range(max_iter_num - 1):
            if iter_num == 0:
                # 第一次迭代开始允许 scale 做多步优化
                self.quantizer.iters_for_scale = max(iters_for_scale, 2)
                _, _, scale_out0, zero_out0, err0 = self.quantizer.find_params(
                    weight_matrix, groupsize=group_size, H=hessian
                )
            else:
                scale_out0, zero_out0 = scale_out, zero_out

            s0, z0 = expand_group_params(weight_matrix, group_size, scale_out0, zero_out0)
            h00, w00 = H00.clone(), W00.clone()
            if actorder:
                h00 = h00[perm][:, perm]
                s0, z0 = s0[:, perm], z0[:, perm]
                w00 = w00[:, perm]

            # 使用 optimize_int_weight_v3 在当前 scale 下优化 int 权重
            w_int0, _, err0 = optimize_int_weight_v3(
                s0,
                z0,
                h00,
                is_symmetric,
                self.quantizer.min_bound,
                self.quantizer.max_bound,
                w00,
                x_init=None,
                max_iter=iters_before_round,
                lr=learning_rate,
                max_inner_iter=max_inner_iter,
                round_fn=round_fn,
            )

            if actorder:
                w_int0 = w_int0[:, invperm]

            if (err0 < 0).any():
                eig = torch.linalg.eigh(H00)
                print("The eigenvalues is", eig.eigenvalues)
                print("The err is ", err0)
                print("The negative err is ", err0[err0 < 0])
                # from IPython import embed; embed(header="negative loss 0")
                # raise ValueError(f"Fatal error, the eigenvalues of hessian is {eig.eigenvalues}")

            better_mask = err0 < err
            w_int[better_mask] = w_int0[better_mask]
            scale_out[better_mask] = scale_out0[better_mask]
            zero_out[better_mask] = zero_out0[better_mask]
            err[better_mask] = err0[better_mask]
            success_rate = torch.sum(better_mask) / better_mask.shape[0]
            print(
                f"Iter {iter_num}, the success rate of opt_intW is {success_rate.item()}, the loss is {err.mean().item()}"
            )
            if success_rate < 1e-4:
                break

            del w_int0, err0, h00, w00, s0, z0, better_mask, success_rate

            # analytic_scale: 给定当前 w_int，再重新求一遍最优 scale/zero
            try:
                scale_out0, zero_out0, err0 = self.quantizer.get_scale_and_zero_out_group(
                    H=H00, groupsize=group_size, x0=W00, x_int=w_int
                )
                if (err0 <= 0).any():
                    eig = torch.linalg.eigh(H00)
                    print(eig.eigenvalues)
                    raise ValueError(
                        f"Fatal error, the eigenvalues os hessian is {eig.eigenvalues}"
                    )
            except torch.cuda.OutOfMemoryError:
                print(
                    "catch an OutOfMemoryError, the shape of the weight is ",
                    W00.shape,
                    "we will spilt to get scale",
                )
                num_part = 16
                num_channel = W00.shape[0]
                ps = num_channel // num_part
                if num_channel % num_part != 0:
                    break
                try:
                    scale_out0s, zero_out0s, err0s = [], [], []
                    # OOM 处理：沿着 out_channel 维度切分
                    for k in range(num_part):
                        scale_out0, zero_out0, err0 = self.quantizer.get_scale_and_zero_out_group(
                            H=H00,
                            groupsize=group_size,
                            x0=W00[k * ps : (k + 1) * ps],
                            x_int=w_int[k * ps : (k + 1) * ps],
                        )
                        scale_out0s.append(scale_out0)
                        zero_out0s.append(zero_out0)
                        err0s.append(err0)
                except torch.cuda.OutOfMemoryError:
                    print(
                        "catch an OutOfMemoryError again, the shape of the weight is ",
                        W00.shape,
                        "give up",
                    )
                    # 如果还是 OOM，就放弃这一轮 scale 搜索
                    torch.cuda.empty_cache()
                    break
                scale_out0, zero_out0, err0 = (
                    torch.cat(scale_out0s),
                    torch.cat(zero_out0s),
                    torch.cat(err0s),
                )

            better_mask = err0 < err
            err[better_mask] = err0[better_mask]
            scale_out[better_mask] = scale_out0[better_mask]
            zero_out[better_mask] = zero_out0[better_mask]
            success_rate = torch.sum(better_mask) / better_mask.shape[0]
            print(
                f"Iter {iter_num}, the success rate of analytic_scale is {success_rate.item()}, the loss is {err.mean().item()}"
            )
            if success_rate < 1e-4:
                break

            del scale_out0, zero_out0, err0, better_mask, success_rate

        print(f"after {iter_num} of pure_training, the loss is {err.mean().item()}")

        # 将 group-wise 的 scale/zero 展开到每个通道
        scale_out0, zero_out0 = expand_group_params(
            weight_matrix, group_size, scale_out, zero_out
        )
        if is_symmetric:
            Q = w_int * scale_out0
        else:
            Q = w_int * scale_out0 + zero_out0

        loss = torch.matmul(torch.matmul((W00 - Q), hessian), (W00 - Q).t()).diag()
        print("finally the loss is ", loss.mean().item())

        # Conv1D 权重需要转回原来的布局
        if self.target_layer.__class__.__name__ in ("Conv1D",) and Q.ndim == 2:
            Q = Q.t()
            w_int = w_int.t()
            scale_out = scale_out.t()
            zero_out = zero_out.t()

        w_int = w_int.reshape(self.target_layer.weight.shape).to(torch.int8)
        self.target_layer.weight.data = Q.reshape(self.target_layer.weight.shape).to(
            self.target_layer.weight.data.dtype
        )
        return scale_out, zero_out, w_int, err

    def free(self):
        """
        释放 Hessian 统计，回收显存
        """
        self.hessian_matrix = None
        torch.cuda.empty_cache()


def override_forward_methods(root_module):
    """
    暂时替换 Linear / Conv1D / Conv2d 的 forward：
    - 在 forward 中显式使用 scale / zero 还原近似 FP 权重
    """
    original_forward_fns = {}
    for module_name, sub_module in root_module.named_modules():
        if isinstance(sub_module, torch.nn.Linear) or sub_module.__class__.__name__ in ("Conv1D",):
            print(f"replace forward for {module_name}")
            original_forward_fns[module_name] = sub_module.forward
            sub_module.forward = linear_forward_with_quant(sub_module)
        elif isinstance(sub_module, (torch.nn.Conv2d,)):
            print(f"replace forward for {module_name}")
            original_forward_fns[module_name] = sub_module.forward
            sub_module.forward = conv2d_forward_with_quant(sub_module)

    return original_forward_fns


def restore_forward_methods(root_module, original_forward_fns):
    """
    恢复被 override 的 forward 函数
    """
    for module_name, sub_module in root_module.named_modules():
        if module_name in original_forward_fns:
            print(f"recover forward for {module_name}")
            sub_module.forward = original_forward_fns[module_name]


def linear_forward_with_quant(layer_module):
    """
    构造带有量化恢复逻辑的 Linear / Conv1D forward
    """

    def _forward(inputs, *args, **kwargs):
        # 这里只处理 Linear / Conv1D（Conv1D 在其他地方已经统一成 Linear 风格）
        if isinstance(layer_module, (torch.nn.Linear,)):
            size_out = inputs.size()[:-1] + (layer_module.weight.shape[0],)
            dim = 1
        else:
            raise NotImplementedError("Fatal Error")

        # 如果有 scale / zero，则按 group_size 重复并恢复近似 FP 权重
        if hasattr(layer_module, "scale"):
            if layer_module.group_size == -1:
                layer_module.group_size = layer_module.weight.shape[dim]
            scale = torch.repeat_interleave(
                layer_module.scale, repeats=layer_module.group_size, dim=dim
            )
            zero = torch.repeat_interleave(
                layer_module.zero, repeats=layer_module.group_size, dim=dim
            )
            weight = layer_module.weight * scale + zero
        else:
            weight = layer_module.weight

        out = F.linear(inputs, weight)
        if layer_module.bias is not None:
            out = out + layer_module.bias
        out = out.view(size_out)
        return out

    return _forward


def conv2d_forward_with_quant(layer_module):
    """
    构造带有量化恢复逻辑的 Conv2d forward
    """

    def _forward(inputs, *args, **kwargs):
        # Conv2d：scale/zero 按输出通道广播
        shape = [layer_module.weight.shape[0]] + (len(layer_module.weight.shape) - 1) * [1]
        if hasattr(layer_module, "scale"):
            scale = torch.reshape(layer_module.scale, shape=shape)
            zero = torch.reshape(layer_module.zero, shape=shape)
            weight = layer_module.weight * scale + zero
        else:
            weight = layer_module.weight

        out = F.conv2d(
            inputs,
            weight,
            layer_module.bias,
            layer_module.stride,
            layer_module.padding,
            layer_module.dilation,
            layer_module.groups,
        )
        return out

    return _forward


@torch.enable_grad()
def finetune_block_scales(
    args, quantizers, layer_block, cached_inputs, device, layer_idx, masks,dir_name
):
    """
    针对某一层 block：
    - 把量化权重替换到网络中
    - 把 scale / zero 设为可训练参数
    - 用离线缓存的输入输出做 MSE 微调（blockwise minimize）
    """
    layer_block = layer_block.to(device)
    sub_layers = collect_quant_layers(layer_block)

    optim_params = []
    original_dtypes = {}

    # 1) 初始化每个子层的权重 & scale / zero 参数
    for key in sub_layers:
        layer_qconfig = quantizers[f"{layer_idx}.{key}.weight"]
        quant_weight = layer_qconfig["weights"]
        scale_list = layer_qconfig["scales"]

        original_dtypes[key] = sub_layers[key].weight.dtype

        dtype = torch.float32
        factory_kwargs = {"device": device, "dtype": dtype}
        sub_layers[key].weight.data = quant_weight.to(**factory_kwargs)
        sub_layers[key].weight.requires_grad_(False)

        scale_param = torch.nn.Parameter(
            scale_list[0].clone().to(**factory_kwargs), requires_grad=True
        )
        requires_grad_zero = True if args.asym else False
        zero_param = torch.nn.Parameter(
            scale_list[1].clone().to(**factory_kwargs), requires_grad=requires_grad_zero
        )

        sub_layers[key].register_parameter("scale", scale_param)
        sub_layers[key].register_parameter("zero", zero_param)
        sub_layers[key].group_size = args.group_size

        optim_params.append(scale_param)
        if args.asym:
            optim_params.append(zero_param)

    # 2) 可选：把 LN / BN 的参数也一并训练
    if args.train_LN:
        for name, module in layer_block.named_modules():
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)) or "Norm" in module.__class__.__name__:
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(True)
                    optim_params.append(module.weight)
                    print("add layer norm weight to train")
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad_(True)
                    optim_params.append(module.bias)
                    print("add layer norm bias to train")

    # 3) 可选：训练 Linear 的 bias
    if args.train_bias:
        for name, module in layer_block.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad_(True)
                    optim_params.append(module.bias)
                    print("add linear bias to train")

    # 暂时替换 forward，让其在前向中使用 (scale, zero)
    original_forward_fns = override_forward_methods(layer_block)

    lr = args.blockwise_minimize_lr
    optimizer = torch.optim.Adam(
        optim_params,
        lr,
        eps=2.0e-5,
        betas=(0.9, 0.99),
        weight_decay=args.blockwise_minimize_wd,
    )
    print("--", optimizer.param_groups[0]["lr"])

    total_loss = 0.0
    for epoch in range(args.blockwise_minimize_epoch):
        for idx, batch_data in enumerate(cached_inputs):
            # batch_data 是 (args, kwargs) 的结构
            batch_data = cast_fp16_to_float32(move_to_device(batch_data, device))
            label = torch.load(f"./{dir_name}/out_{idx}.pth")
            mask = masks[idx]

            # 前向传播，使用量化权重 + 可训练的 scale/zero
            out = layer_block(*(batch_data[0]), **(batch_data[1]))
            res = out[0] - move_to_device(label["out"][0], device)

            # 如果有 mask，按 token 掩码计算 MSE
            if mask is not None:
                res = res * mask.float().unsqueeze(-1)
                loss = torch.sum(res * res) / (mask.float().sum() * res.shape[-1])
            else:
                loss = torch.mean(res * res)

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        print(f"the avg loss for training scale zero is {total_loss / len(cached_inputs)}")
        total_loss = 0.0

    # 训练结束后，把 scale/zero 写回 quantizers，并恢复真正的权重
    for key in sub_layers:
        scale_param, zero_param = sub_layers[key].scale, sub_layers[key].zero
        quantizers[f"{layer_idx}.{key}.weight"]["scales"] = [scale_param.cpu(), zero_param.cpu()]

        # 不同层类型，对应的 group 维度不同
        if sub_layers[key].__class__.__name__ in ("Conv1D",):
            dim = 0
        elif isinstance(sub_layers[key], torch.nn.Linear):
            dim = 1
        else:
            raise NotImplementedError

        group_size = args.group_size
        if group_size == -1:
            group_size = sub_layers[key].weight.data.shape[dim]

        # 展开 group-wise 的 scale/zero 到每个通道
        scale_full = torch.repeat_interleave(scale_param, repeats=group_size, dim=dim)
        zero_full = torch.repeat_interleave(zero_param, repeats=group_size, dim=dim)

        sub_layers[key].weight.data = sub_layers[key].weight.data * scale_full + zero_full
        sub_layers[key].weight.data = sub_layers[key].weight.data.to(original_dtypes[key])

        # 清理临时属性
        del sub_layers[key].scale, sub_layers[key].zero, sub_layers[key].group_size

    # 恢复原始 forward
    restore_forward_methods(layer_block, original_forward_fns)
    print()





