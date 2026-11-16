# motion-vector-generator

Python 游戏渲染用的逐像素屏幕空间运动向量生成器。输入当前帧深度、前后两帧
view-projection 矩阵、分辨率、子像素抖动和动态物体变换，输出每像素屏幕运动向量
与无效掩码，可直接驱动时域抗锯齿（TAA）、运动模糊和速度场调试。

## 安装

依赖仅为 NumPy：

```bash
python3 -m pip install numpy
```

## 快速开始

```python
import numpy as np
from motion_vectors import generate_motion_vectors, DynamicObject

# depth: (height, width)，当前帧 NDC z 深度；空像素写 NaN，天空写 1.0
result = generate_motion_vectors(
    depth=depth,
    view_proj=view_proj_curr,
    view_proj_prev=view_proj_prev,
    resolution=(width, height),
    jitter_curr=(0.25, -0.15),   # 单位：像素，y 向下为正
    jitter_prev=(0.0, 0.0),
    dynamic_objects=[
        DynamicObject(object_id=7,
                      model_prev=model_prev,
                      model_curr=model_curr),
    ],
    object_ids=object_id_buffer,  # -1 表示静态几何
)

result.motion  # (height, width, 2)，单位像素，(dx, dy)，dy 向下为正
result.mask    # (height, width) uint8，INVALID_* 位域
```

动态像素先用 `inverse(model_curr)` 回到当前物体空间，再用 `model_prev` 重新
表达为上一帧世界位置，然后才走相机重投影，因此角色与相机同时移动时仍能得到
正确方向的拖影，而不是被相机运动抵消成零向量。

## 坐标约定

### NDC

- 采用 OpenGL 风格的齐次裁剪空间 / NDC：裁剪点 `(cx, cy, cz, cw)` 除以 `cw`
  得到 NDC。
- NDC `x, y ∈ [-1, 1]`，y 轴朝上；NDC `z ∈ [-1, 1]`，近平面映射到 `-1`，
  远平面映射到 `+1`。
- 输入深度即 NDC z。`NaN`、正负无穷以及 `empty_value` 哨兵视为空深度；
  深度达到远平面（`z ≈ +1`）视为天空。
- 上一帧重投影后若 `w_prev <= 0`（点落在上一帧相机后方），或 NDC z 超出
  `[-1, 1]`（近/远裁剪），该像素标记为 `INVALID_BEHIND_CAMERA`，运动向量置零。

### 像素中心

- 像素坐标以左上角像素为原点，列向右、行向下。
- 像素 `(col, row)` 的中心为
  `ndc_x = (2*col + 1)/width - 1`、
  `ndc_y = 1 - (2*row + 1)/height`。
- 几何重投影始终在**未抖动的像素中心**进行；子像素抖动以像素为单位、围绕
  像素中心施加，y 向下为正。
- 最终运动向量显式叠加抖动差 `jitter_prev - jitter_curr`，因此抖动的贡献与
  相机几何运动解耦，便于单独掩码与调试。

### 上一帧坐标约定

- 运动向量单位为像素，方向从**当前采样点指向上一帧采样点**
  （即 TAA / 运动模糊回查历史时的行走方向）：
  `motion = previous_sample - current_sample`。
- `motion[..., 0]` 为水平分量（右正左负），`motion[..., 1]` 为垂直分量
  （下正上负）。
- 上一帧采样点超出视口时额外标记 `INVALID_OUT_OF_SCREEN`，向量仍保留有限值。
- 所有输出保证有限：相机后方、近裁剪、空深度、天空等无法重投影的像素向量
  置零，绝不上溢成极大运动向量。

## 无效掩码

`mask` 是 uint8 位域，可组合多个标志：

| 标志 | 含义 |
| --- | --- |
| `INVALID_EMPTY` | 空深度（NaN / 无穷 / 哨兵值），无表面可重建 |
| `INVALID_BEHIND_CAMERA` | 重投影点位于上一帧相机后方，或被近/远平面裁剪 |
| `INVALID_DEPTH_BOUNDARY` | 与四邻域的线性眼空间深度跳变超过阈值（跨深度边界） |
| `INVALID_JITTER` | 两帧之间存在非零子像素抖动差 |
| `INVALID_OUT_OF_SCREEN` | 上一帧采样点落在视口之外 |
| `INVALID_SKY` | 深度位于远平面（天空），强制零向量而非无界向量 |

深度边界使用当前帧相机空间 `w`（线性眼深度）做相对比较
`|d - d_neighbour| / max(d, eps) > threshold`，空 / 天空邻域也视为不连续，
阈值由 `depth_boundary_threshold` 配置（默认 0.1）。

## 运行测试

```bash
python3 -m unittest
```

测试覆盖相机平移与旋转的重投影方向、相机后方点、近裁剪面、远景天空、动态
角色、子像素抖动、跨深度边界、空深度和所有输出的有限性。
