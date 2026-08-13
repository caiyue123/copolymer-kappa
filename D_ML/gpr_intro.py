# 高斯过程回归（GPR）入门 —— 角色 D 的第一课
# 运行方法（在 PowerShell 里）:
#   wsl -e /home/ldeath/mlenv/bin/python /mnt/c/1234/D_ML/gpr_intro.py
# 或者先进 wsl 再运行:
#   source ~/mlenv/bin/activate
#   python /mnt/c/1234/D_ML/gpr_intro.py
#
# 这个例子和你们项目的对应关系:
#   x  = 序列特征 (交替度、嵌段长度...)      -> 这里是 1 维演示用 sin 函数
#   y  = 热导率 κ                            -> 这里是 sin(x) + 噪声
#   GPR 输出: 后验均值(预测值) + 方差(不确定性) -> 主动学习靠不确定性来推荐新序列

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# ---------- 中文字体设置 (借用 Windows 自带字体, 避免图里中文变方框) ----------
import matplotlib.font_manager as fm
for _f in ['/mnt/c/Windows/Fonts/simhei.ttf', '/mnt/c/Windows/Fonts/msyh.ttc']:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False
avail = {f.name for f in fm.fontManager.ttflist}
for _name in ['SimHei', 'Microsoft YaHei']:
    if _name in avail:
        plt.rcParams['font.sans-serif'] = [_name, 'DejaVu Sans']
        break

# ---------- 1. 造假数据: 12 个样本, 模拟"初始 5 个训练序列" ----------
rng = np.random.RandomState(42)
X = rng.uniform(0, 10, 12)[:, None]                 # 样本点 (特征)
y = np.sin(X[:, 0]) + 0.1 * rng.randn(12)           # 目标值 (热导率)

# ---------- 2. 训练 GPR ----------
# 核函数 = 常数项 * RBF(径向基); length_scale 是"相关性衰减距离", 是核心超参数
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
gp.fit(X, y)

# ---------- 3. 预测: 均值 + 标准差 ----------
X_test = np.linspace(0, 10, 200)[:, None]
y_mean, y_std = gp.predict(X_test, return_std=True)

# ---------- 4. 画图 ----------
plt.figure(figsize=(9, 5))
plt.plot(X_test, y_mean, 'b-', label='后验均值 (预测)')
plt.fill_between(X_test[:, 0], y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                 alpha=0.2, color='b', label='95% 置信区间 (不确定性)')
plt.scatter(X[:, 0], y, c='r', s=40, label='样本点 (已模拟的序列)')
plt.xlabel('x (序列特征)'); plt.ylabel('y (热导率)')
plt.title('GPR 第一课: 预测 + 不确定性 = 主动学习的地基')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('/mnt/c/1234/D_ML/gpr_intro.png', dpi=150)
plt.show() if False else None   # 无图形界面时只存图

print('学到的核函数超参数:', gp.kernel_)
print('前 3 个测试点的预测均值:', np.round(y_mean[:3], 3))
print('前 3 个测试点的标准差  :', np.round(y_std[:3], 3))
print('图片已保存: C:\\1234\\D_ML\\gpr_intro.png')
print()
print('思考题: 标准差大的地方说明什么? 主动学习(EI)会优先推荐那里吗?')
