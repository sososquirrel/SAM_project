import matplotlib as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Polygon, Rectangle
from scipy import interpolate


def associated_rectangle(
    center: np.array,
    length: float,
    width: float,
    orientation_radian: float,
    orientation_degrees: float,
):

    x_prev, y_prev = (
        center[0]
        - (width / 2 * np.cos(orientation_radian) + length / 2 * np.sin(orientation_radian)),
        center[1]
        - (-width / 2 * np.sin(orientation_radian) + length / 2 * np.cos(orientation_radian)),
    )

    rectangle = patches.Rectangle(
        (x_prev, y_prev),
        width,
        length,
        edgecolor="k",
        fill=False,
        lw=2,
        angle=-orientation_degrees,
    )

    return rectangle


def get_coords_rectangle(rectangle):

    # coords = getattr(rectangle, get_patch_transform().transform(r1.get_path().vertices[:-1]))

    arg = getattr(getattr(rectangle, "get_path")(), "vertices")[:-1]
    coords = getattr(getattr(rectangle, "get_patch_transform")(), "transform")(arg)

    return coords


def from_points_to_segment(pointA, pointB, n_points: int = 40):
    xx = np.linspace(pointA[0], pointB[0], n_points)
    yy = np.linspace(pointA[1], pointB[1], n_points)
    return (xx, yy)


def get_rectangle_sides_points(rectangle):

    coords = get_coords_rectangle(rectangle)

    xx1, yy1 = from_points_to_segment(coords[0], coords[3])

    xx2, yy2 = from_points_to_segment(coords[1], coords[2])

    xx3, yy3 = from_points_to_segment(coords[3], coords[2])

    xx4, yy4 = from_points_to_segment(coords[0], coords[1])

    slides = [[xx1, yy1], [xx2, yy2], [xx3, yy3], [xx4, yy4]]

    return slides


def get_rectangle_top_surface_points(rectangle):

    coords = get_coords_rectangle(rectangle)
    slides = get_rectangle_sides_points(rectangle)

    X_top_surface_points, Y_top_surface_points = np.array([]), np.array([])
    XX1, YY1 = slides[0]
    XX2, YY2 = slides[1]

    n_points = len(slides[0][0])

    for i in range(n_points):
        coord_i = [[XX1[i], YY1[i]], [XX2[i], YY2[i]]]
        x_seg, y_seg = from_points_to_segment(coord_i[0], coord_i[1])

        X_top_surface_points, Y_top_surface_points = (
            np.concatenate((X_top_surface_points, x_seg)),
            np.concatenate((Y_top_surface_points, y_seg)),
        )

    return X_top_surface_points, Y_top_surface_points


def get_horizontally_interpolated_fields(
    U_fields_3D: np.array, V_fields_3D: np.array, W_fields_3D: np.array
):

    U_interpolate = []
    V_interpolate = []
    W_interpolate = []

    nz, ny, nx = U_fields_3D.shape

    x = np.arange(0, nx, 1)
    y = np.arange(0, ny, 1)

    for i in range(nz):
        f1 = interpolate.interp2d(y, x, U_fields_3D[i], kind="cubic")
        f2 = interpolate.interp2d(y, x, V_fields_3D[i], kind="cubic")
        f3 = interpolate.interp2d(y, x, W_fields_3D[i], kind="cubic")

        U_interpolate.append(f1)
        V_interpolate.append(f2)
        W_interpolate.append(f3)

    return U_interpolate, V_interpolate, W_interpolate


def evaluate_fields_over_rectangle_face(U_interpolate, V_interpolate, W_interpolate, rectangle):

    [[xx1, yy1], [xx2, yy2], [xx3, yy3], [xx4, yy4]] = get_rectangle_sides_points(rectangle)

    X_top_surface_points, Y_top_surface_points = get_rectangle_top_surface_points(rectangle)

    n_points = len(xx1)

    u_surface_1 = [[u(xx1[i], yy1[i])[0] for i in range(n_points)] for u in U_interpolate]
    v_surface_1 = [[v(xx1[i], yy1[i])[0] for i in range(n_points)] for v in V_interpolate]
    u_surface_2 = [[u(xx2[i], yy2[i])[0] for i in range(n_points)] for u in U_interpolate]
    v_surface_2 = [[v(xx2[i], yy2[i])[0] for i in range(n_points)] for v in V_interpolate]
    u_surface_3 = [[u(xx3[i], yy3[i])[0] for i in range(n_points)] for u in U_interpolate]
    v_surface_3 = [[v(xx3[i], yy3[i])[0] for i in range(n_points)] for v in V_interpolate]
    u_surface_4 = [[u(xx4[i], yy4[i])[0] for i in range(n_points)] for u in U_interpolate]
    v_surface_4 = [[v(xx4[i], yy4[i])[0] for i in range(n_points)] for v in V_interpolate]

    # staggered_grid_W = 0.5 * (np.array(W_interpolate[-2]) + np.array(W_interpolate[-1]))

    w_surface_5 = [
        [
            [w(X_top_surface_points[i], Y_top_surface_points[i])[0]]
            for i in range(len(X_top_surface_points))
        ]
        for w in W_interpolate[-3:-1]
    ]

    w_surface_5 = np.array(w_surface_5)

    w_surface_5 = np.mean(w_surface_5, axis=0)

    w_surface_5_reshape = np.reshape(w_surface_5, (n_points, n_points))

    return (
        np.array(u_surface_1),
        np.array(v_surface_1),
        np.array(u_surface_2),
        np.array(v_surface_2),
        np.array(u_surface_3),
        np.array(v_surface_3),
        np.array(u_surface_4),
        np.array(v_surface_4),
        w_surface_5_reshape,
    )


def surface_differential(rectangle, vertical_array):

    [[xx1, yy1], [xx2, yy2], [xx3, yy3], [xx4, yy4]] = get_rectangle_sides_points(rectangle)

    width, height = getattr(rectangle, "get_width")(), getattr(rectangle, "get_height")()

    nz = len(vertical_array)
    n_points = len(xx1)

    dydz = np.zeros((nz, n_points))
    for i in range(nz):
        for j in range(n_points):
            dydz[i, j] = height / n_points * np.gradient(vertical_array / 1000)[i]

    dxdz = np.zeros((nz, n_points))
    for i in range(nz):
        for j in range(n_points):
            dxdz[i, j] = width / n_points * np.gradient(vertical_array / 1000)[i]

    dxdy = width / n_points * height / n_points * np.ones((n_points, n_points))

    return dydz, dxdz, dxdy


def get_surface_flux_side(U_surface, V_surface, dS, angle_radian, direction):

    if direction == "length":
        flux = np.sum(
            ((U_surface * np.cos(angle_radian) - V_surface * np.sin(angle_radian)))[:-1, :]
            * dS[:-1, :]
        )

    elif direction == "width":

        flux = np.sum(
            ((U_surface * np.sin(angle_radian) + V_surface * np.cos(angle_radian)))[:-1, :]
            * dS[:-1, :]
        )

    return flux


def get_surface_flux_top(W_surface, dxdy):

    flux = np.sum(W_surface * dxdy)

    return flux


def mass_balance(
    center: np.array,
    length: float,
    width: float,
    orientation_radian: float,
    orientation_degrees: float,
    U_fields_3D: np.array,
    V_fields_3D: np.array,
    W_fields_3D: np.array,
    vertical_array: np.array,
):

    rectangle = associated_rectangle(
        center=center,
        length=length,
        width=width,
        orientation_radian=orientation_radian,
        orientation_degrees=orientation_degrees,
    )

    coords = get_coords_rectangle(rectangle=rectangle)

    U_interpolate, V_interpolate, W_interpolate = get_horizontally_interpolated_fields(
        U_fields_3D=U_fields_3D, V_fields_3D=V_fields_3D, W_fields_3D=W_fields_3D
    )

    (
        u_surface_1,
        v_surface_1,
        u_surface_2,
        v_surface_2,
        u_surface_3,
        v_surface_3,
        u_surface_4,
        v_surface_4,
        w_surface_5,
    ) = evaluate_fields_over_rectangle_face(
        U_interpolate, V_interpolate, W_interpolate, rectangle
    )

    dydz, dxdz, dxdy = surface_differential(rectangle=rectangle, vertical_array=vertical_array)

    print(np.sum(dydz), np.sum(dxdz), np.sum(dxdy))

    u_flux_1 = get_surface_flux_side(
        U_surface=u_surface_1,
        V_surface=v_surface_1,
        dS=dydz,
        angle_radian=orientation_radian,
        direction="length",
    )

    u_flux_2 = get_surface_flux_side(
        U_surface=u_surface_2,
        V_surface=v_surface_2,
        dS=dydz,
        angle_radian=orientation_radian,
        direction="length",
    )

    u_flux_3 = get_surface_flux_side(
        U_surface=u_surface_3,
        V_surface=v_surface_3,
        dS=dxdz,
        angle_radian=orientation_radian,
        direction="width",
    )

    u_flux_4 = get_surface_flux_side(
        U_surface=u_surface_4,
        V_surface=v_surface_4,
        dS=dxdz,
        angle_radian=orientation_radian,
        direction="width",
    )

    u_flux_5 = get_surface_flux_top(W_surface=w_surface_5, dxdy=dxdy)

    return (
        u_flux_1,
        u_flux_2,
        u_flux_3,
        u_flux_4,
        u_flux_5,
    )
