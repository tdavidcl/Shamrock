"""
FMM math demo in python
=======================

This example shows how to use the FMM maths to compute the force between two points
"""

# %%
# As always, we start by importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import shamrock

# %%
# Utilities
# ^^^^^^^^^
# You can ignore this first block, it just contains some utility functions to draw the AABB and the arrows
# We only defines the function `draw_aabb` and `draw_arrow`, which are used to draw the AABB and the arrows in the plots
# and the function `draw_box_pair`, which is used to draw the box pair with all the vectors needed to compute the FMM force

# %%
# .. raw:: html
#
#   <details>
#   <summary><a>Click here to expand the utility code</a></summary>
#



def draw_aabb(ax, aabb, color, alpha):
    """
    Draw a 3D AABB in matplotlib

    Parameters
    ----------
    ax : matplotlib.Axes3D
        The axis to draw the AABB on
    aabb : shamrock.math.AABB_f64_3
        The AABB to draw
    color : str
        The color of the AABB
    alpha : float
        The transparency of the AABB
    """
    xmin, ymin, zmin = aabb.lower
    xmax, ymax, zmax = aabb.upper

    points = [
        aabb.lower,
        (aabb.lower[0], aabb.lower[1], aabb.upper[2]),
        (aabb.lower[0], aabb.upper[1], aabb.lower[2]),
        (aabb.lower[0], aabb.upper[1], aabb.upper[2]),
        (aabb.upper[0], aabb.lower[1], aabb.lower[2]),
        (aabb.upper[0], aabb.lower[1], aabb.upper[2]),
        (aabb.upper[0], aabb.upper[1], aabb.lower[2]),
        aabb.upper,
    ]

    faces = [
        [points[0], points[1], points[3], points[2]],
        [points[4], points[5], points[7], points[6]],
        [points[0], points[1], points[5], points[4]],
        [points[2], points[3], points[7], points[6]],
        [points[0], points[2], points[6], points[4]],
        [points[1], points[3], points[7], points[5]],
    ]

    edges = [
        [points[0], points[1]],
        [points[0], points[2]],
        [points[0], points[4]],
        [points[1], points[3]],
        [points[1], points[5]],
        [points[2], points[3]],
        [points[2], points[6]],
        [points[3], points[7]],
        [points[4], points[5]],
        [points[4], points[6]],
        [points[5], points[7]],
        [points[6], points[7]],
    ]

    collection = Poly3DCollection(faces, alpha=alpha, color=color)
    ax.add_collection3d(collection)

    edge_collection = Line3DCollection(edges, color="k", alpha=alpha)
    ax.add_collection3d(edge_collection)


def draw_arrow(ax, p1, p2, color, label, arr_scale=0.2):
    length = np.linalg.norm(np.array(p2) - np.array(p1))
    arrow_length_ratio = arr_scale / length
    ax.quiver(
        p1[0],
        p1[1],
        p1[2],
        p2[0] - p1[0],
        p2[1] - p1[1],
        p2[2] - p1[2],
        color=color,
        label=label,
        arrow_length_ratio=arrow_length_ratio,
    )


def draw_box_pair(ax, box_1_center, box_2_center, box_1_size, box_2_size):

    box_1 = shamrock.math.AABB_f64_3(
        (
            box_1_center[0] - box_1_size / 2,
            box_1_center[1] - box_1_size / 2,
            box_1_center[2] - box_1_size / 2,
        ),
        (
            box_1_center[0] + box_1_size / 2,
            box_1_center[1] + box_1_size / 2,
            box_1_center[2] + box_1_size / 2,
        ),
    )
    box_2 = shamrock.math.AABB_f64_3(
        (
            box_2_center[0] - box_2_size / 2,
            box_2_center[1] - box_2_size / 2,
            box_2_center[2] - box_2_size / 2,
        ),
        (
            box_2_center[0] + box_2_size / 2,
            box_2_center[1] + box_2_size / 2,
            box_2_center[2] + box_2_size / 2,
        ),
    )

    ax.scatter(p_i[0], p_i[1], p_i[2], color="red", label="p_i")
    ax.scatter(p_j[0], p_j[1], p_j[2], color="blue", label="p_j")

    ax.scatter(
        box_1_center[0], box_1_center[1], box_1_center[2], color="black", label="box_1_center"
    )
    ax.scatter(
        box_2_center[0], box_2_center[1], box_2_center[2], color="black", label="box_2_center"
    )

    draw_aabb(ax, box_1, "b", 0.1)
    draw_aabb(ax, box_2, "r", 0.1)

    center_view = (1.0, 0.0, 0.0)
    view_size = 3.0
    ax.set_xlim(center_view[0] - view_size / 2, center_view[0] + view_size / 2)
    ax.set_ylim(center_view[1] - view_size / 2, center_view[1] + view_size / 2)
    ax.set_zlim(center_view[2] - view_size / 2, center_view[2] + view_size / 2)

    # arrow from p_i to center of box_1
    draw_arrow(ax, box_1_center, p_i, "black", "p_i <- box_1")
    draw_arrow(ax, box_2_center, p_j, "black", "p_j <- box_2")

    # arrow from center of box_1 to center of box_2
    draw_arrow(ax, box_1_center, box_2_center, "black", "box_1 -> box_2")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


# %%
# .. raw:: html
#
#   </details>


# %%
# 
# FMM force computation
# ^^^^^^^^^^^^^^^^^^^^^
#
# Let's start by assuming that we have two particles at positions :math:`\mathbf{x}_i` and
# :math:`\mathbf{x}_j` contained in two boxes (:math:`A` and :math:`B`) whose centers are at positions
# :math:`\mathbf{s}_a` and :math:`\mathbf{s}_b` respectively.
# The positions of the particles relative to their respective boxes are then:
#
# .. math::
#    \mathbf{a}_i = \mathbf{x}_i - \mathbf{s}_a \\
#    \mathbf{b}_j = \mathbf{x}_j - \mathbf{s}_b
#
# and the distance between the centers of the boxes is:
#
# .. math::
#    \mathbf{r} = \mathbf{s}_b - \mathbf{s}_a
#
# This implies that the distance between the two particles is:
#
# .. math::
#    \mathbf{x}_j - \mathbf{x}_i = \mathbf{r} + \mathbf{b}_j - \mathbf{a}_i
#
# If we denote the Green function for an inverse distance :math:`G(\mathbf{x}) = 1 / \vert\vert\mathbf{x}\vert\vert`, then the potential exerted onto particle :math:`i` is:
#
# .. math::
#    \Phi_i = \Phi (\mathbf{x}_i) &= \int  - \frac{\mathcal{G} \rho(\mathbf{x}_j)}{\vert\vert\mathbf{x}_i - \mathbf{x}_j\vert\vert} d\mathbf{x}_j \\
#    &= - \mathcal{G} \int \rho(\mathbf{x}_j) G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j
#
# and the force exerted onto particle :math:`i` is:
#
# .. math::
#    \mathbf{f}_i = -\nabla_i \Phi (\mathbf{x}_i) &= \int - \nabla_i \frac{\mathcal{G} \rho(\mathbf{x}_j)}{\vert\vert\mathbf{x}_i - \mathbf{x}_j\vert\vert} d\mathbf{x}_j \\
#    &= \mathcal{G} \int \rho(\mathbf{x}_j) \nabla_i G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j \\
#    &= -\mathcal{G} \int \rho(\mathbf{x}_j) \nabla_j G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j
#
# Now let's expand the green function in a Taylor series to order :math:`p`.
#
# .. math::
#    G(\mathbf{x}_j - \mathbf{x}_i) &= G(\mathbf{r} + \mathbf{b}_j - \mathbf{a}_i) \\
#       &= \sum_{k = 0}^p \frac{(-1)^k}{k!} \mathbf{a}_i^{(k)} \cdot \sum_{n=0}^{p-k} \frac{1}{n!} D_{n+k} \cdot \mathbf{b}_j^{(n)}
#
# where :math:`D_{n} = \nabla^{(n)}_r G(\mathbf{r})` is the n-th order derivative of the Green
# function and the operator :math:`\mathbf{a}_i^{(k)}` is the tensor product of :math:`k` :math:`\mathbf{a}_i` vectors.

# %%
#
# Similarly the gradient of the green function is:
#
# .. math::
#    \nabla_j G(\mathbf{x}_j - \mathbf{x}_i) &= \nabla_r G(\mathbf{r} + \mathbf{b}_j - \mathbf{a}_i) \\
#       &= \sum_{k = 0}^p \frac{(-1)^k}{k!} \mathbf{a}_i^{(k)} \cdot \sum_{n=0}^{p-k} \frac{1}{n!} D_{n+k+1} \cdot \mathbf{b}_j^{(n)}
#
# Now we can plug that back into the expression for the force & potential:
#
# .. math::
#    \Phi_i &= - \mathcal{G} \int \rho(\mathbf{x}_j) G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j \\
#    &= - \mathcal{G} \sum_{k = 0}^p \frac{(-1)^k}{k!} \mathbf{a}_i^{(k)} \cdot \underbrace{\sum_{n=0}^{p-k} \frac{1}{n!} D_{n+k} \cdot \underbrace{\int \rho(\mathbf{x}_j) \mathbf{b}_j^{(n)} d\mathbf{x}_j}_{Q_n^B}}_{M_{p,k}} \\
#
# .. math::
#    \mathbf{f}_i &= -\mathcal{G} \int \rho(\mathbf{x}_j) \nabla_j G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j \\
#    &= -\mathcal{G}  \sum_{k = 0}^p \frac{(-1)^k}{k!} \mathbf{a}_i^{(k)} \cdot \underbrace{\sum_{n=0}^{p-k} \frac{1}{n!} D_{n+k+1} \cdot  \underbrace{\int \rho(\mathbf{x}_j)\mathbf{b}_j^{(n)} d\mathbf{x}_j}_{Q_n^B}}_{dM_{p,k} = M_{p+1,k+1}}
#

# %%
#
# As one can tell sadly the two expressions while similar do not share the same terms.
# 
# I will not go in this rabit hole of using the same expansion for both now but the idea is to 
# use the primitive of the force which is the same expansion as the force but with the primitive 
# of :math:`\mathbf{a}_i^{(k)}` instead.
#
# .. math::
#    \Phi_i  = - \int \mathbf{f}_i =  -\mathcal{G}  \sum_{k = 0}^p \frac{(-1)^k}{k!} \int\mathbf{a}_i^{(k)} \cdot {M_{p+1,k+1}}
#   

# %%
# 
# Mass moments
# ^^^^^^^^^^^^


# %%
# This is a helper to plot the situation
def plot_mass_moment_case(box_B_center,box_B_size,x_j):
    box_B = shamrock.math.AABB_f64_3(
            (
                box_B_center[0] - box_B_size / 2,
                box_B_center[1] - box_B_size / 2,
                box_B_center[2] - box_B_size / 2,
            ),
            (
                box_B_center[0] + box_B_size / 2,
                box_B_center[1] + box_B_size / 2,
                box_B_center[2] + box_B_size / 2,
            ),
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_arrow(ax, box_B_center, x_j, "black", "$x_j - s_B$")

    ax.scatter(
            box_B_center[0], box_B_center[1], box_B_center[2], color="black", label="box_B_center"
        )

    ax.scatter(
        x_j[0], x_j[1], x_j[2], color="red", label="$x_j$"
    )

    draw_aabb(ax, box_B, "blue",0.2)

    center_view = (0.0, 0.0, 0.0)
    view_size = 2.0
    ax.set_xlim(center_view[0] - view_size / 2, center_view[0] + view_size / 2)
    ax.set_ylim(center_view[1] - view_size / 2, center_view[1] + view_size / 2)
    ax.set_zlim(center_view[2] - view_size / 2, center_view[2] + view_size / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax



# %%
# Let's start with the following 

s_B = (0,0,0)
box_B_size = 1

x_j = (0.2,0.2,0.2)
m_j = 1

b_j = (x_j[0] - s_B[0],x_j[1] - s_B[1],x_j[2] - s_B[2])

ax = plot_mass_moment_case(s_B,box_B_size,x_j)
plt.title(f"Mass moment illustration")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()


# %%
# Here the mass moment of a set of particles (here only one) of mass :math:`m_j` is
#
# .. math::
#    {Q_n^B} &= \int \rho(\mathbf{x}_j) \mathbf{b}_j^{(n)} d\mathbf{x}_j\\
#            &= \sum_j m_j \mathbf{b}_j^{(n)}
# 
# In Shamrock python bindings the function 

# %%
# .. code-block::
# 
#        shamrock.math.SymTensorCollection_f64_<low order>_<high order>.from_vec(b_j)

# %%
# will return the collection of symetrical tensors :math:`\mathbf{b}_j^{(n)}` for n in between `<low order>` and `<high order>`
# Here are the values of the tensors :math:`{Q_n^B}` from order 0 up to 5 using shamrock symmetrical tensor collections

Q_n_B = shamrock.math.SymTensorCollection_f64_0_5.from_vec(b_j)
Q_n_B *= m_j

print(Q_n_B)

# %%
# Now if we take a displacment that is only along the x axis we get null components in the Q_n_B if for cases that do not only exhibit x

s_B = (0,0,0)
box_B_size = 1

x_j = (0.5,0.0,0.0)
m_j = 1

b_j = (x_j[0] - s_B[0],x_j[1] - s_B[1],x_j[2] - s_B[2])

ax = plot_mass_moment_case(s_B,box_B_size,x_j)
plt.title(f"Mass moment illustration")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()

Q_n_B = shamrock.math.SymTensorCollection_f64_0_5.from_vec(b_j)
Q_n_B *= m_j

print(Q_n_B)


# %%
# 
# Gravitational moments
# ^^^^^^^^^^^^^^^^^^^^^

# %%

def run_fmm(p_i, p_j, box_1_center, box_2_center, order=4, do_print=False):

    # %%
    # Compute the mass moment of box_1
    delta_1 = (p_i[0] - box_1_center[0], p_i[1] - box_1_center[1], p_i[2] - box_1_center[2])
    delta_r = (
        box_1_center[0] - box_2_center[0],
        box_1_center[1] - box_2_center[1],
        box_1_center[2] - box_2_center[2],
    )
    delta_2_j = (p_j[0] - box_2_center[0], p_j[1] - box_2_center[1], p_j[2] - box_2_center[2])

    # such that p_i - p_j = delta_1 + delta_r - delta_2_j

    if do_print:
        print("delta_1 =", delta_1)

    if order == 0:
        Q_n_1 = shamrock.math.SymTensorCollection_f64_0_0.from_vec(delta_1)
    elif order == 1:
        Q_n_1 = shamrock.math.SymTensorCollection_f64_0_1.from_vec(delta_1)
    elif order == 2:
        Q_n_1 = shamrock.math.SymTensorCollection_f64_0_2.from_vec(delta_1)
    elif order == 3:
        Q_n_1 = shamrock.math.SymTensorCollection_f64_0_3.from_vec(delta_1)
    elif order == 4:
        Q_n_1 = shamrock.math.SymTensorCollection_f64_0_4.from_vec(delta_1)
    elif order == 5:
        Q_n_1 = shamrock.math.SymTensorCollection_f64_0_5.from_vec(delta_1)
    else:
        raise ValueError("Invalid order")

    Q_n_1 *= m_i
    if do_print:
        print("Q_n_1 =", Q_n_1)

    # %%
    # Green function from box_1 to box_2
    # delta_r = (box_2_center[0] - box_1_center[0], box_2_center[1] - box_1_center[1], box_2_center[2] - box_1_center[2])

    if order == 0:
        D_n = shamrock.phys.green_func_grav_cartesian_1_1(delta_r)
    elif order == 1:
        D_n = shamrock.phys.green_func_grav_cartesian_1_2(delta_r)
    elif order == 2:
        D_n = shamrock.phys.green_func_grav_cartesian_1_3(delta_r)
    elif order == 3:
        D_n = shamrock.phys.green_func_grav_cartesian_1_4(delta_r)
    elif order == 4:
        D_n = shamrock.phys.green_func_grav_cartesian_1_5(delta_r)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("D_n =", D_n)

    D_n *= -1

    # %%
    # Gravitational moment of box_1 on box_2
    if order == 0:
        dM_n = shamrock.phys.get_dM_mat_0(D_n, Q_n_1)
    elif order == 1:
        dM_n = shamrock.phys.get_dM_mat_1(D_n, Q_n_1)
    elif order == 2:
        dM_n = shamrock.phys.get_dM_mat_2(D_n, Q_n_1)
    elif order == 3:
        dM_n = shamrock.phys.get_dM_mat_3(D_n, Q_n_1)
    elif order == 4:
        dM_n = shamrock.phys.get_dM_mat_4(D_n, Q_n_1)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("dM_n =", dM_n)

    # %%
    # Local displacement moment of p_j in box_2
    if order == 0:
        Aj_n_2 = shamrock.math.SymTensorCollection_f64_0_0.from_vec(delta_2_j)
    elif order == 1:
        Aj_n_2 = shamrock.math.SymTensorCollection_f64_0_1.from_vec(delta_2_j)
    elif order == 2:
        Aj_n_2 = shamrock.math.SymTensorCollection_f64_0_2.from_vec(delta_2_j)
    elif order == 3:
        Aj_n_2 = shamrock.math.SymTensorCollection_f64_0_3.from_vec(delta_2_j)
    elif order == 4:
        Aj_n_2 = shamrock.math.SymTensorCollection_f64_0_4.from_vec(delta_2_j)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("Aj_n_2 =", Aj_n_2)

    # %%
    # FMM force on p_j
    if order == 0:
        force_j = shamrock.phys.contract_grav_moment_to_force_1(Aj_n_2, dM_n)
    elif order == 1:
        force_j = shamrock.phys.contract_grav_moment_to_force_2(Aj_n_2, dM_n)
    elif order == 2:
        force_j = shamrock.phys.contract_grav_moment_to_force_3(Aj_n_2, dM_n)
    elif order == 3:
        force_j = shamrock.phys.contract_grav_moment_to_force_4(Aj_n_2, dM_n)
    elif order == 4:
        force_j = shamrock.phys.contract_grav_moment_to_force_5(Aj_n_2, dM_n)
    else:
        raise ValueError("Invalid order")

    return (force_j[0], force_j[1], force_j[2])


def analytic_force_j(p_i, p_j, box_1_center, box_2_center):
    force_j_direct = (p_i[0] - p_j[0], p_i[1] - p_j[1], p_i[2] - p_j[2])
    force_j_direct /= np.linalg.norm(force_j_direct) ** 3
    force_j_direct *= m_i
    return force_j_direct


# %%
# A first example
m_i = 1.0
m_j = 1.0

p_i = (0.0, 0.0, 0.0)
p_j = (2.0, 0.0, 0.0)

box_1_center = (0.0, 0.5, 0.0)
box_2_center = (2.0, 0.5, 0.2)

box_1_size = 1.0
box_2_size = 1.0

order = 4

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

force_j = run_fmm(p_i, p_j, box_1_center, box_2_center, order=order, do_print=True)

draw_box_pair(ax, box_1_center, box_2_center, box_1_size, box_2_size)

force_j_direct = analytic_force_j(p_i, p_j, box_1_center, box_2_center)

p_i = np.array(p_i)
p_j = np.array(p_j)
force_j = np.array(force_j)
force_j_direct = np.array(force_j_direct)

print("force_j =", force_j)
print("force_j_direct =", force_j_direct)
abs_error = np.linalg.norm(force_j - force_j_direct)
rel_error = abs_error / np.linalg.norm(force_j_direct)
print("abs error =", abs_error)
print("rel error =", rel_error)

draw_arrow(ax, p_j, p_j + force_j * 3, "green", "force_j")
draw_arrow(ax, p_j, p_j + force_j_direct * 3, "red", "force_j_direct")

plt.title(f"FMM order = {order}, rel error = {rel_error}")

plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))

plt.show()


# %%
# Sweeping through FMM orders
# sphinx_gallery_multi_image = "single"

m_i = 1.0
m_j = 1.0

p_i = (0.0, 0.0, 0.0)
p_j = (2.0, 0.0, 0.0)

box_1_center = (0.0, 0.5, 0.0)
box_2_center = (2.0, 0.5, 0.2)

box_1_size = 1.0
box_2_size = 1.0


for order in range(1, 4):
    print("--------------------------------")
    print(f"Running FMM order = {order}")
    print("--------------------------------")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    force_j = run_fmm(p_i, p_j, box_1_center, box_2_center, order=order, do_print=True)

    draw_box_pair(ax, box_1_center, box_2_center, box_1_size, box_2_size)

    # Direct force on p_j for comparison
    force_j_direct = analytic_force_j(p_i, p_j, box_1_center, box_2_center)

    p_i = np.array(p_i)
    p_j = np.array(p_j)
    force_j = np.array(force_j)
    force_j_direct = np.array(force_j_direct)

    print("force_j =", force_j)
    print("force_j_direct =", force_j_direct)
    abs_error = np.linalg.norm(force_j - force_j_direct)
    rel_error = abs_error / np.linalg.norm(force_j_direct)
    print("abs error =", abs_error)
    print("rel error =", rel_error)

    draw_arrow(ax, p_j, p_j + force_j * 3, "green", "force_j")
    draw_arrow(ax, p_j, p_j + force_j_direct * 3, "red", "force_j_direct")

    plt.title(f"FMM order = {order}, rel error = {rel_error}")

    plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))


plt.show()


# %%
# Sweeping through opening angles
m_i = 1.0
m_j = 1.0

p_i_rel = (0.0, -0.5, 0.0)
p_j = (2.0, 0.0, 0.0)

box_1_centers = [(-1.0, 0.5, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0)]
box_2_center = (2.0, 0.5, 0.2)

box_1_size = 1.0
box_2_size = 1.0
order = 4

for box_1_center in box_1_centers:
    print("--------------------------------")
    print(f"Running FMM order = {order} for box_1_center = {box_1_center}")
    print("--------------------------------")

    p_i = (p_i_rel[0] + box_1_center[0], p_i_rel[1] + box_1_center[1], p_i_rel[2] + box_1_center[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    force_j = run_fmm(p_i, p_j, box_1_center, box_2_center, order=order, do_print=True)

    draw_box_pair(ax, box_1_center, box_2_center, box_1_size, box_2_size)

    # Direct force on p_j for comparison
    force_j_direct = analytic_force_j(p_i, p_j, box_1_center, box_2_center)

    p_i = np.array(p_i)
    p_j = np.array(p_j)
    force_j = np.array(force_j)
    force_j_direct = np.array(force_j_direct)

    print("force_j =", force_j)
    print("force_j_direct =", force_j_direct)
    abs_error = np.linalg.norm(force_j - force_j_direct)
    rel_error = abs_error / np.linalg.norm(force_j_direct)
    print("abs error =", abs_error)
    print("rel error =", rel_error)

    draw_arrow(ax, p_j, p_j + force_j * 3, "green", "force_j")
    draw_arrow(ax, p_j, p_j + force_j_direct * 3, "red", "force_j_direct")

    plt.title(f"FMM order = {order}, rel error = {rel_error}")

    plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))


plt.show()


# %%
# Testing the precision

plt.figure()
for order in range(1, 5):
    print("--------------------------------")
    print(f"Running FMM order = {order}")
    print("--------------------------------")

    # set seed
    np.random.seed(111)

    N = 1000

    # generate a random set of position in a box of bounds (-1,1)x(-1,1)x(-1,1)
    p_i = []
    for i in range(N):
        p_i.append((np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)))

    # same for p_j
    p_j = []
    for i in range(N):
        p_j.append((np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)))

    box_scale_fact = 0.01

    # same for box_1_center
    box_1_centers = []
    for p in p_i:
        box_1_centers.append(
            (
                p[0] + box_scale_fact * np.random.uniform(-1, 1),
                p[1] + box_scale_fact * np.random.uniform(-1, 1),
                p[2] + box_scale_fact * np.random.uniform(-1, 1),
            )
        )

    # same for box_2_center
    box_2_centers = []
    for p in p_j:
        box_2_centers.append(
            (
                p[0] + box_scale_fact * np.random.uniform(-1, 1),
                p[1] + box_scale_fact * np.random.uniform(-1, 1),
                p[2] + box_scale_fact * np.random.uniform(-1, 1),
            )
        )

    angles = []
    rel_errors = []

    for p_i, p_j, box_1_center, box_2_center in zip(p_i, p_j, box_1_centers, box_2_centers):

        # print(f"p_i = {p_i}")
        # print(f"p_j = {p_j}")
        # print(f"box_1_center = {box_1_center}")
        # print(f"box_2_center = {box_2_center}")

        force_j = run_fmm(p_i, p_j, box_1_center, box_2_center, order=order, do_print=False)
        force_j_direct = analytic_force_j(p_i, p_j, box_1_center, box_2_center)
        abs_error = np.linalg.norm(force_j - force_j_direct)
        rel_error = abs_error / np.linalg.norm(force_j_direct)

        p_i = np.array(p_i)
        p_j = np.array(p_j)
        box_1_center = np.array(box_1_center)
        box_2_center = np.array(box_2_center)
        force_j = np.array(force_j)
        force_j_direct = np.array(force_j_direct)

        b_i_size = np.linalg.norm(box_1_center - p_i)
        b_j_size = np.linalg.norm(box_2_center - p_j)

        b_dist = np.linalg.norm(box_1_center - box_2_center)

        angle = (b_i_size + b_j_size) / b_dist

        if angle > 1.0:
            continue

        angles.append(angle)
        rel_errors.append(rel_error)

        print(f"angle = {angle}, rel error = {rel_error}")

    plt.plot(angles, rel_errors, "o", label=f"FMM order = {order}")

plt.xlabel("Angle")
plt.ylabel("Relative Error")
plt.xscale("log")
plt.yscale("log")
plt.title("FMM precision")
plt.legend()
plt.grid()
plt.show()
