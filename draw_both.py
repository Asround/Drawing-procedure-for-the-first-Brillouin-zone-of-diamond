# -*- coding: utf-8 -*-
"""
交互式绘制（Plotly）：
1) 金刚石晶胞：红=顶点+面心；蓝=内部四原子；边界实线；可选虚线"键"
2) 第一布里渊区（截角八面体）：直接几何求顶点与面，只绘外表面；新增"棱线"覆盖
3) 倒格子 BCC 常规立方体（仅画一个立方体的边界棱；可开关）
4) 可将交互式图保存为本地 HTML（完全离线、单文件）：参数 save_html
5) 左图显示原胞基矢箭头 a₁,a₂,a₃：参数 show_primitive_vectors
6) 右图显示特殊点 L/X/K/Γ：参数 show_special
"""

import os
import numpy as np
from itertools import product, combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
# 截角八面体（第一布里渊区）几何求解
# ---------------------------

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
# 倒格子 BCC 常规立方体（仅棱）
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


# ---------------------------
# 作图（左右并排）
# ---------------------------

def plot_diamond_and_bz(a=1.0,
                        show_dashed_bonds=True,
                        show_bz_edges=True,
                        bz_edge_width=4,
                        # 保存交互式 HTML 的文件路径（None 表示不保存）
                        save_html=None,
                        # 是否绘制倒格子 BCC 常规立方体（仅棱）
                        show_bcc_cell=False,
                        bcc_cell_width=3,
                        bcc_cell_color='gray',
                        # （新增）左图原胞基矢箭头
                        show_primitive_vectors=False,
                        primitive_color='darkgreen',
                        primitive_width=6,
                        primitive_cone_scale=0.08,  # 箭头锥体大小 ~ 0.08*a
                        # （新增）右图显示特殊点 L/X/K/Γ
                        show_special=True,
                        special_point_size=5,  # 特殊点大小
                        point_size=6):
    # ==== 左图：金刚石晶胞 ====
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

    # ==== 右图：第一布里渊区（截角八面体） ====
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

    # ---------------------------
    # 组合子图
    # ---------------------------
    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        horizontal_spacing=0.05,
        subplot_titles=("金刚石晶胞<br>Diamond Unit Cell",
                        "第一布里渊区（截角八面体；单位：2π/a）<br>First Brillouin Zone (truncated octahedron; unit: 2π/a)")
    )

    # 调整子图标题位置（下移）
    fig.update_annotations(
        yshift=-30  # 负值表示下移，单位是像素
    )

    # 左：红点分为顶点和面心
    # 顶点
    fig.add_trace(go.Scatter3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        mode='markers', marker=dict(size=point_size, color='red'),
        name='顶点',
        hovertemplate='<b>晶格点(顶点)</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
    ), row=1, col=1)
    # 面心
    fig.add_trace(go.Scatter3d(
        x=face_centers[:, 0], y=face_centers[:, 1], z=face_centers[:, 2],
        mode='markers', marker=dict(size=point_size, color='red'),
        name='面心',
        hovertemplate='<b>晶格点(面心)</b><br>x: %{x:.2f><br>y: %{y:.2f><br>z: %{z:.2f><extra></extra>',
    ), row=1, col=1)

    # 蓝点
    fig.add_trace(go.Scatter3d(
        x=blue_pts[:, 0], y=blue_pts[:, 1], z=blue_pts[:, 2],
        mode='markers', marker=dict(size=point_size, color='blue'),
        name='蓝：内部四原子',
        hovertemplate='<b>内部原子</b><br>x: %{x:.2f><br>y: %{y:.2f><br>z: %{z:.2f><extra></extra>',
    ), row=1, col=1)

    # 边界
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(width=3, color='black'),
        name='晶胞边界',
        hoverinfo='none',
    ), row=1, col=1)

    # 虚线键
    if show_dashed_bonds:
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(width=2, color='blue', dash='dash'),
            name='内部原子-最近邻(虚线)',
            hoverinfo='none',  # 禁用悬停显示
        ), row=1, col=1)

    # 左：原胞基矢箭头 a1,a2,a3（锥体放在面心端点）
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
            hovertemplate='<b>原胞基矢a₁</b><extra></extra>',
        ), row=1, col=1)

        # a2 基矢
        fig.add_trace(go.Scatter3d(
            x=[0, a2[0]], y=[0, a2[1]], z=[0, a2[2]],
            mode='lines',
            line=dict(width=primitive_width, color=primitive_color),
            name='原胞基矢a₂',
            hovertemplate='<b>原胞基矢a₂</b><extra></extra>',
        ), row=1, col=1)

        # a3 基矢
        fig.add_trace(go.Scatter3d(
            x=[0, a3[0]], y=[0, a3[1]], z=[0, a3[2]],
            mode='lines',
            line=dict(width=primitive_width, color=primitive_color),
            name='原胞基矢a₃                ',
            hovertemplate='<b>原胞基矢a₃</b><extra></extra>',
        ), row=1, col=1)

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
            hoverinfo='skip',
            showlegend=False  # 不显示在图例中
        ), row=1, col=1)

        # 文本标签：略微超出端点，避免与锥体重叠
        label_pts = np.vstack([a1, a2, a3]) * 1.08
        fig.add_trace(go.Scatter3d(
            x=label_pts[:, 0], y=label_pts[:, 1], z=label_pts[:, 2],
            mode='text',
            text=['a₁', 'a₂', 'a₃'],
            textposition='middle right',
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

    # 右：截角八面体网格（仅外表面）
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=I, j=J, k=K,
        opacity=0.35, color='gold',
        flatshading=True,
        name='第一布里渊区（截角八面体）',
        hoverinfo='skip'
    ), row=1, col=2)

    # 右：棱线（截角八面体）
    if show_bz_edges:
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode='lines',
            line=dict(width=bz_edge_width, color='black'),
            name='截角八面体棱',
            hoverinfo='skip'
        ), row=1, col=2)

    # 右：倒格 BCC 常规立方体棱
    if show_bcc_cell:
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode='lines',
            line=dict(width=bcc_cell_width, color='gray'),
            name='倒格子 BCC 常规立方体（仅棱）',
            hoverinfo='skip'
        ), row=1, col=2)

    # 右：Γ 点与邻点 + 虚线参考
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(size=point_size + 2, color='red'),
        name='Γ（原点）',
        hovertemplate=special_hover_info['Γ'] + '<extra></extra>'
    ), row=1, col=2)

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
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode='lines',
        line=dict(width=2, color='blue', dash='dash'),
        name='Γ-邻点连线(虚线)',
        hovertemplate='<b>Γ点到倒易点连线</b><extra></extra>'
    ), row=1, col=2)

    # 右：特殊点（L/X/K/Γ）和标签
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
        ), row=1, col=2)

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
        ), row=1, col=2)

    # 统一视图/坐标轴样式
    scene_common = dict(
        xaxis=dict(title='x', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        yaxis=dict(title='y', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        zaxis=dict(title='z', showspikes=False, backgroundcolor='rgba(240,240,240,0.2)'),
        aspectmode='cube'
    )

    # 更新布局，控制图例项之间的间隔
    fig.update_layout(
        scene=scene_common,
        scene2=dict(**scene_common, xaxis_title='k_x (2π/a)', yaxis_title='k_y (2π/a)', zaxis_title='k_z (2π/a)'),
        # 设置图例位置和样式
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,  # 图例在子图标题下方
            xanchor="center",
            x=0.5,
            tracegroupgap=1,  # 增加图例项之间的间隔（像素）
            itemsizing='constant',
            font=dict(
                size=10,  # 调整图例文本大小
                family="Arial"
            ),
            itemwidth=30,  # 图例项的宽度（像素）- 必须 >= 30
            bordercolor="LightGray",
            borderwidth=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=0, r=0, t=20, b=120)  # 增加底部边距以容纳图例
    )

    # === 保存交互式 HTML（完全离线、单文件）===
    if save_html is not None:
        save_dir = os.path.dirname(save_html)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.write_html(save_html, include_plotlyjs='inline', include_mathjax=False, full_html=True)

    fig.show()


# 运行示例
if __name__ == "__main__":
    plot_diamond_and_bz(
        a=1.0,  # 晶格常数
        show_dashed_bonds=True,  # 金刚石晶胞内部连线(第二套晶格)
        show_bz_edges=True,  # 截角八面体的棱
        bz_edge_width=4,  # 截角八面体棱的粗细
        show_bcc_cell=True,  # 倒格 BCC 常规立方体（仅棱）
        # 新增：显示原胞基矢箭头
        show_primitive_vectors=True,  # 是否显示原胞基矢箭头
        primitive_color='darkgreen',  # 原胞基矢箭头颜色
        primitive_width=6,  # 原胞基矢线粗细
        primitive_cone_scale=0.08,  # 原胞基矢箭头大小
        # 新增：显示特殊点
        show_special=True,  # 是否显示特殊点
        special_point_size=5,  # 特殊点大小
        # 保存为完全离线 HTML, 直接可以打开, 无需其他支持
        save_html="outputs/diamond_bz.html", # 输出路径和文件名
        point_size=6
    )