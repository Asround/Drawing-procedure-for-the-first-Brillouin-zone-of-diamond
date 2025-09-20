# -*- coding: utf-8 -*-
"""
独立绘制第一布里渊区（截角八面体）
"""
import os
import numpy as np
from itertools import product, combinations
import plotly.graph_objects as go


# ---------------------------
# 工具函数
# ---------------------------

def bcc_conventional_cube_edges(a=1.0):
    """
    以 Γ 为体心，边长 a* = 4π/a 的 BCC 常规立方体。
    角点在 (±a*/2, ±a*/2, ±a*/2) = (±2π/a, ±2π/a, ±2π/a)。
    返回用于 Plotly 折线的 x,y,z（以 None 分段）
    """
    g = 2 * np.pi / a  # a*/2
    corners = np.array(list(product([-g, g], repeat=3)), dtype=float)  # 8 角点
    xs, ys, zs = [], [], []
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(np.linalg.norm(corners[i] - corners[j]) - 2 * g) < 1e-9:
                xs += [corners[i, 0], corners[j, 0], None]
                ys += [corners[i, 1], corners[j, 1], None]
                zs += [corners[i, 2], corners[j, 2], None]
    return xs, ys, zs


def truncated_octahedron_vertices(a=1.0):
    gscale = 2 * np.pi / a
    constraints = []

    # 8 个 ±<111>
    for s in product([-1, 1], repeat=3):
        G = gscale * np.array(s, dtype=float)
        b = 0.5 * np.dot(G, G)
        constraints.append((G, b))

    # 6 个 ±<200>
    for axis in [(2, 0, 0), (0, 2, 0), (0, 0, 2)]:
        for s in [-1, 1]:
            v = np.array(axis, dtype=float) * s
            G = gscale * v
            b = 0.5 * np.dot(G, G)
            constraints.append((G, b))

    # 去重
    uniq, seen = [], set()
    for A, b in constraints:
        key = tuple(np.round(A, 12)) + (round(b, 12),)
        if key not in seen:
            uniq.append((A, b));
            seen.add(key)
    constraints = uniq  # 14

    # 求所有三平面交点，筛半空间
    verts = []
    for (A1, b1), (A2, b2), (A3, b3) in combinations(constraints, 3):
        A = np.vstack([A1, A2, A3])
        det = np.linalg.det(A)
        if abs(det) < 1e-9:
            continue
        p = np.linalg.solve(A, np.array([b1, b2, b3]))
        if all(np.dot(Aq, p) - bq <= 1e-7 for Aq, bq in constraints):
            verts.append(p)

    # 顶点去重
    V, seen = [], set()
    for p in verts:
        key = tuple(np.round(p / gscale, 7))
        if key not in seen:
            seen.add(key);
            V.append(p)
    return np.array(V), constraints


def faces_from_constraints(verts, constraints, tol=1e-7):
    faces = []
    for A, b in constraints:
        idxs = [i for i, p in enumerate(verts) if abs(np.dot(A, p) - b) < 10 * tol]
        if len(idxs) >= 3:
            faces.append((A, b, idxs))
    return faces


def sort_face_vertices_ccw(verts, A, idxs):
    n = A / np.linalg.norm(A)
    P = verts[idxs]
    c = P.mean(axis=0)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref);
    u /= np.linalg.norm(u)
    v = np.cross(n, u);
    v /= np.linalg.norm(v)
    angs = []
    for i in idxs:
        vec = verts[i] - c
        x = np.dot(vec, u);
        y = np.dot(vec, v)
        angs.append(np.arctan2(y, x))
    order = [i for _, i in sorted(zip(angs, idxs))]
    return order


def triangulate_faces(orders):
    I = [];
    J = [];
    K = []
    for order in orders:
        n = len(order)
        for t in range(1, n - 1):
            I.append(order[0]);
            J.append(order[t]);
            K.append(order[t + 1])
    return np.array(I), np.array(J), np.array(K)


def unique_edges_from_orders(orders):
    edge_set = set();
    edges = []
    for order in orders:
        n = len(order)
        for i in range(n):
            a = order[i];
            b = order[(i + 1) % n]
            key = tuple(sorted((a, b)))
            if key not in edge_set:
                edge_set.add(key);
                edges.append((a, b))
    return edges


# ---------------------------
# 绘制第一布里渊区
# ---------------------------

def plot_brillouin_zone(a=1.0,
                        show_bz_edges=True,
                        bz_edge_width=4,
                        show_bcc_cell=False,
                        bcc_cell_width=3,
                        bcc_cell_color='gray',
                        show_special=True,
                        special_point_size=5,
                        point_size=6,
                        save_html="outputs/brillouin_zone.html"):
    # 确保输出目录存在
    if save_html:
        os.makedirs(os.path.dirname(save_html), exist_ok=True)

    # 第一布里渊区数据
    verts, constraints = truncated_octahedron_vertices(a=a)
    faces = faces_from_constraints(verts, constraints)
    orders = [sort_face_vertices_ccw(verts, A, idxs) for (A, b, idxs) in faces]
    I, J, K = triangulate_faces(orders)

    # 生成"棱线"坐标（仅外表面）
    edge_x, edge_y, edge_z = [], [], []
    if show_bz_edges:
        edges = unique_edges_from_orders(orders)
        for a_idx, b_idx in edges:
            pa, pb = verts[a_idx], verts[b_idx]
            edge_x += [pa[0], pb[0], None]
            edge_y += [pa[1], pb[1], None]
            edge_z += [pa[2], pb[2], None]

    # Γ 与近邻/次近邻倒易点及虚线
    gscale = 2 * np.pi / a
    G111 = [gscale * np.array(s, dtype=float) for s in product([-1, 1], repeat=3)]
    G200 = [gscale * np.array(v, float) for v in [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)]]
    gx, gy, gz = [], [], []
    for G in (G111 + G200):
        gx += [0, G[0], None];
        gy += [0, G[1], None];
        gz += [0, G[2], None]

    # 倒格 BCC 常规立方体棱
    if show_bcc_cell:
        cx, cy, cz = bcc_conventional_cube_edges(a=a)
    else:
        cx, cy, cz = [], [], []

    # 特殊点坐标（单位：2π/a）
    gscale = 2 * np.pi / a

    special_points = {
        'Γ': np.array([0, 0, 0]),
        'X': np.array([1 * gscale, 0, 0]),
        'L': np.array([0.5 * gscale, 0.5 * gscale, 0.5 * gscale]),
        'K': np.array([0.75 * gscale, 0.75 * gscale, 0])
    }

    # 特殊点标签位置（稍微偏移以避免重叠）
    label_offset = 0.1
    special_labels_pos = {
        'Γ': special_points['Γ'] + np.array([label_offset, label_offset, label_offset]),
        'X': special_points['X'] + np.array([label_offset, 0, 0]),
        'L': special_points['L'] + np.array([label_offset, label_offset, label_offset]),
        'K': special_points['K'] + np.array([label_offset, label_offset, 0])
    }

    # 特殊点的悬停信息
    special_hover_info = {
        'Γ': 'Γ点 (0,0,0)<br>布里渊区中心',
        'X': 'X点 (1,0,0)<br>布里渊区边沿与<100>轴的交点',
        'L': 'L点 (0.5,0.5,0.5)<br>布里渊区边沿与<111>轴的交点',
        'K': 'K点 (0.75,0.75,0)<br>布里渊区边沿与<110>轴的交点'
    }

    # 创建图形
    fig = go.Figure()

    # 截角八面体网格（仅外表面）
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=I, j=J, k=K,
        opacity=0.35, color='gold',
        flatshading=True,
        name='第一布里渊区（截角八面体）',
        hoverinfo='skip'
    ))

    # 棱线（截角八面体）
    if show_bz_edges:
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode='lines',
            line=dict(width=bz_edge_width, color='black'),
            name='截角八面体棱',
            hoverinfo='skip'
        ))

    # 倒格 BCC 常规立方体棱
    if show_bcc_cell:
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode='lines',
            line=dict(width=bcc_cell_width, color='gray'),
            name='倒格子 BCC 常规立方体（仅棱）',
            hoverinfo='skip'
        ))

    # Γ 点与邻点 + 虚线参考
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(size=point_size + 2, color='red'),
        name='Γ（原点）',
        hovertemplate=special_hover_info['Γ'] + '<extra></extra>'
    ))

    Gs_disp = np.array(G111 + G200)
    # 为倒易点添加悬停信息
    hover_texts = []
    for G in Gs_disp:
        # 判断是111还是200类型
        if np.count_nonzero(np.abs(G) == gscale) == 3:  # 三个分量都不为零，是111类型
            hover_texts.append(
                f"倒易点 ({G[0] / gscale:.0f},{G[1] / gscale:.0f},{G[2] / gscale:.0f})<br>类型: 最近邻倒格点")
        else:  # 200类型
            hover_texts.append(
                f"倒易点 ({G[0] / gscale:.0f},{G[1] / gscale:.0f},{G[2] / gscale:.0f})<br>类型: 次近邻倒格点")

    fig.add_trace(go.Scatter3d(
        x=Gs_disp[:, 0], y=Gs_disp[:, 1], z=Gs_disp[:, 2],
        mode='markers', marker=dict(size=point_size, color='blue'),
        name='近邻/次近邻倒易点',
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>'
    ))

    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode='lines',
        line=dict(width=2, color='blue', dash='dash'),
        name='Γ-邻点连线(虚线)',
        hovertemplate='<b>Γ点到倒易点连线</b><extra></extra>'
    ))

    # 特殊点（L/X/K/Γ）和标签
    if show_special:
        # 特殊点（L/X/K）用绿色小球
        special_coords = [special_points[key] for key in ['X', 'L', 'K']]
        special_names = ['X', 'L', 'K']
        special_hover_texts = [special_hover_info[key] for key in ['X', 'L', 'K']]

        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in special_coords],
            y=[p[1] for p in special_coords],
            z=[p[2] for p in special_coords],
            mode='markers',
            marker=dict(size=special_point_size, color='green'),
            name='特殊点（L/X/K）',
            text=special_hover_texts,
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))

        # 特殊点标签
        fig.add_trace(go.Scatter3d(
            x=[special_labels_pos[key][0] for key in special_labels_pos],
            y=[special_labels_pos[key][1] for key in special_labels_pos],
            z=[special_labels_pos[key][2] for key in special_labels_pos],
            mode='text',
            text=['Γ', 'X', 'L', 'K'],
            textfont=dict(size=14, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # 设置场景
    scene = dict(
        xaxis=dict(title='k_x (2π/a)', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        yaxis=dict(title='k_y (2π/a)', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        zaxis=dict(title='k_z (2π/a)', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        aspectmode='cube'
    )

    # 更新布局，标题居中，图例居右并从上到下排列
    fig.update_layout(
        title=dict(
            text="第一布里渊区（截角八面体；单位：2π/a）<br>First Brillouin Zone (truncated octahedron; unit: 2π/a)",
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
    plot_brillouin_zone(
        a=1.0,  # 晶格常数
        show_bz_edges=True,  # 截角八面体的棱
        bz_edge_width=4,  # 截角八面体棱的粗细
        show_bcc_cell=True,  # 倒格 BCC 常规立方体（仅棱）
        bcc_cell_width=3,  # 立方体棱粗细
        bcc_cell_color='gray',  # 立方体棱颜色
        show_special=True,  # 是否显示特殊点
        special_point_size=5,  # 特殊点大小
        point_size=6,  # 点大小
        save_html="outputs/brillouin_zone.html"  # 保存为HTML文件
    )
