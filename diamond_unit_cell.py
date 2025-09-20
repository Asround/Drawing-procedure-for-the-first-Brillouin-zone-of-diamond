# -*- coding: utf-8 -*-
"""
独立绘制金刚石晶胞
"""
import os
import numpy as np
from itertools import product, combinations
import plotly.graph_objects as go


# ---------------------------
# 工具函数
# ---------------------------

def cube_edges(a=1.0):
    corners = np.array(list(product([0, a], repeat=3)), dtype=float)
    idx = {(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)}
    xs, ys, zs = [], [], []
    for i, j in idx:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]
    return xs, ys, zs


def nearest_bonds_for_internal(a=1.0, tol=1e-9):
    corners = np.array(list(product([0, a], repeat=3)), dtype=float)
    face_centers = np.array([
        [a / 2, a / 2, 0], [a / 2, a / 2, a],
        [a / 2, 0, a / 2], [a / 2, a, a / 2],
        [0, a / 2, a / 2], [a, a / 2, a / 2]
    ], dtype=float)
    red = np.vstack([corners, face_centers])
    blue = np.array([
        [a / 4, a / 4, a / 4],
        [3 * a / 4, 3 * a / 4, a / 4],
        [3 * a / 4, a / 4, 3 * a / 4],
        [a / 4, 3 * a / 4, 3 * a / 4]
    ], dtype=float)
    bonds = []
    target = a * np.sqrt(3) / 4
    for b in blue:
        dists = np.linalg.norm(red - b, axis=1)
        idxs = np.where(np.abs(dists - target) < tol)[0]
        if len(idxs) != 4:
            idxs = np.argsort(np.abs(dists - target))[:4]
        for k in idxs:
            bonds.append((b, red[k]))
    return bonds, red, blue


# ---------------------------
# 绘制金刚石晶胞
# ---------------------------

def plot_diamond_cell(a=1.0,
                      show_dashed_bonds=True,
                      show_primitive_vectors=False,
                      primitive_color='darkgreen',
                      primitive_width=6,
                      primitive_cone_scale=0.08,
                      point_size=6,
                      save_html="outputs/diamond_cell.html"):
    # 确保输出目录存在
    if save_html:
        os.makedirs(os.path.dirname(save_html), exist_ok=True)

    # 金刚石晶胞数据
    corners = np.array(list(product([0, a], repeat=3)), dtype=float)
    face_centers = np.array([
        [a / 2, a / 2, 0], [a / 2, a / 2, a],
        [a / 2, 0, a / 2], [a / 2, a, a / 2],
        [0, a / 2, a / 2], [a, a / 2, a / 2]
    ], dtype=float)
    red_pts = np.vstack([corners, face_centers])
    blue_pts = np.array([
        [a / 4, a / 4, a / 4],
        [3 * a / 4, 3 * a / 4, a / 4],
        [3 * a / 4, a / 4, 3 * a / 4],
        [a / 4, 3 * a / 4, 3 * a / 4]
    ], dtype=float)
    ex, ey, ez = cube_edges(a=a)

    bonds, _, _ = nearest_bonds_for_internal(a=a)
    bx, by, bz = [], [], []
    if show_dashed_bonds:
        for b, r in bonds:
            bx += [b[0], r[0], None]
            by += [b[1], r[1], None]
            bz += [b[2], r[2], None]

    # 创建图形
    fig = go.Figure()

    # 红点分为顶点和面心
    # 顶点
    fig.add_trace(go.Scatter3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        mode='markers', marker=dict(size=point_size, color='red'),
        name='顶点',
        hovertemplate='<b>晶格点(顶点)</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))
    # 面心
    fig.add_trace(go.Scatter3d(
        x=face_centers[:, 0], y=face_centers[:, 1], z=face_centers[:, 2],
        mode='markers', marker=dict(size=point_size, color='red'),
        name='面心',
        hovertemplate='<b>晶格点(面心)</b><br>x: %{x:.2f><br>y: %{y:.2f><br>z: %{z:.2f><extra></extra>'
    ))

    # 蓝点
    fig.add_trace(go.Scatter3d(
        x=blue_pts[:, 0], y=blue_pts[:, 1], z=blue_pts[:, 2],
        mode='markers', marker=dict(size=point_size, color='blue'),
        name='内部四原子',
        hovertemplate='<b>内部原子</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))

    # 边界
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(width=3, color='black'),
        name='晶胞边界',
        # hovertemplate='<b>晶胞边界</b><extra></extra>'
        hoverinfo='none'
    ))

    # 虚线键
    if show_dashed_bonds:
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(width=2, color='blue', dash='dash'),
            name='内部原子-最近邻(虚线)',
            hoverinfo='none'
        ))

    # 原胞基矢箭头 a1,a2,a3（锥体放在面心端点）
    if show_primitive_vectors:
        a1 = np.array([0, a / 2, a / 2], float)
        a2 = np.array([a / 2, 0, a / 2], float)
        a3 = np.array([a / 2, a / 2, 0], float)

        # 分开绘制每个基矢，以便分别悬停
        # a1 基矢
        fig.add_trace(go.Scatter3d(
            x=[0, a1[0]], y=[0, a1[1]], z=[0, a1[2]],
            mode='lines',
            line=dict(width=primitive_width, color=primitive_color),
            name='原胞基矢a₁',
            hovertemplate='<b>原胞基矢a₁</b><extra></extra>'
        ))

        # a2 基矢
        fig.add_trace(go.Scatter3d(
            x=[0, a2[0]], y=[0, a2[1]], z=[0, a2[2]],
            mode='lines',
            line=dict(width=primitive_width, color=primitive_color),
            name='原胞基矢a₂',
            hovertemplate='<b>原胞基矢a₂</b><extra></extra>'
        ))

        # a3 基矢
        fig.add_trace(go.Scatter3d(
            x=[0, a3[0]], y=[0, a3[1]], z=[0, a3[2]],
            mode='lines',
            line=dict(width=primitive_width, color=primitive_color),
            name='原胞基矢a₃',
            hovertemplate='<b>原胞基矢a₃</b><extra></extra>'
        ))

        # 锥体：放在端点(面心)；锥尖在端点，指向原点（即方向取 -a_i）
        fig.add_trace(go.Cone(
            x=[a1[0], a2[0], a3[0]],
            y=[a1[1], a2[1], a3[1]],
            z=[a1[2], a2[2], a3[2]],
            u=[a1[0], a2[0], a3[0]],
            v=[a1[1], a2[1], a3[1]],
            w=[a1[2], a2[2], a3[2]],
            anchor="tip",  # 关键：锥尖在给定坐标
            sizemode="absolute",
            sizeref=max(1e-9, primitive_cone_scale * a),
            showscale=False,
            colorscale=[[0, primitive_color], [1, primitive_color]],
            name='',
            hoverinfo='skip'
        ))

        # 文本标签：略微超出端点，避免与锥体重叠
        label_pts = np.vstack([a1, a2, a3]) * 1.08
        fig.add_trace(go.Scatter3d(
            x=label_pts[:, 0], y=label_pts[:, 1], z=label_pts[:, 2],
            mode='text',
            text=['a₁', 'a₂', 'a₃'],
            textposition='middle right',
            showlegend=False,
            hoverinfo='skip'
        ))

    # 设置场景
    scene = dict(
        xaxis=dict(title='x', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        yaxis=dict(title='y', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        zaxis=dict(title='z', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        aspectmode='cube'
    )

    # 更新布局，标题居中，图例居右并从上到下排列
    fig.update_layout(
        title=dict(
            text="金刚石晶胞<br>Diamond Unit Cell",
            x=0.5,  # 标题居中
            xanchor='center'
        ),
        scene=scene,
        legend=dict(
            orientation="v",  # 垂直排列
            yanchor="top",  # 顶部对齐
            y=0.99,  # 靠近顶部
            xanchor="right",  # 右侧对齐
            x=0.99  # 靠近右侧
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        width=None,  # 设置为None以自适应宽度
        height=800,  # 设置适当的高度
        autosize=True  # 启用自动调整大小
    )

    # 保存HTML
    if save_html:
        fig.write_html(save_html, include_plotlyjs='inline', include_mathjax=False, full_html=True)

    fig.show()


# 运行示例
if __name__ == "__main__":
    plot_diamond_cell(
        a=1.0,  # 晶格常数
        show_dashed_bonds=True,  # 金刚石晶胞内部连线(第二套晶格)
        show_primitive_vectors=True,  # 是否显示原胞基矢箭头
        primitive_color='darkgreen',  # 原胞基矢箭头颜色
        primitive_width=6,  # 原胞基矢线粗细
        primitive_cone_scale=0.08,  # 原胞基矢箭头大小
        point_size=6,  # 点大小
        save_html="outputs/diamond_cell.html"  # 保存为HTML文件
    )

# # 运行示例
# if __name__ == "__main__":
#     plot_diamond_and_bz(
#         a=1.0,  # 晶格常数
#         show_dashed_bonds=True,  # 金刚石晶胞内部连线(第二套晶格)
#         show_bz_edges=True,  # 截角八面体的棱
#         bz_edge_width=4,  # 截角八面体棱的粗细
#         show_bcc_cell=True,  # 倒格 BCC 常规立方体（仅棱）
#         # 新增：显示原胞基矢箭头
#         show_primitive_vectors=True,  # 是否显示原胞基矢箭头
#         primitive_color='darkgreen',  # 原胞基矢箭头颜色
#         primitive_width=6,  # 原胞基矢线粗细
#         primitive_cone_scale=0.08,  # 原胞基矢箭头大小
#         # 新增：显示特殊点
#         show_special=True,  # 是否显示特殊点
#         special_point_size=5,  # 特殊点大小
#         # 保存为完全离线 HTML, 直接可以打开, 无需其他支持
#         save_html="outputs/diamond_bz.html",
#         point_size=6
#     )
