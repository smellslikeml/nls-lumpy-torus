"""Triangulated Laplace-Beltrami — the extension past surfaces of revolution.

The rest of nls_torus assembles the metric ds^2 = dx^2 + A(x)^2 dtheta^2, which can only
express genus-1, rotationally-symmetric shapes. This module carries the same physics onto
an ARBITRARY triangle mesh via the cotangent (FEM) Laplacian, unlocking genuinely new
topology and geometry: genus-2 (two handles), and — with an absorbing collar — non-compact
horns. The operator is validated against the round-sphere spectrum l(l+1) before any
science is done on it.

Core:
    cotan_laplacian(V, F) -> (W, M)   stiffness (PSD, ~ -Delta) + lumped mass (diagonal)
    laplace_spectrum(V, F, k)         smallest k eigenpairs of  W phi = lambda M phi
Generators:
    icosphere(n)                      round unit sphere (validation: eig ~ l(l+1))
    two_tori_genus2(...)              connected genus-2 surface (marching cubes)
    euler_genus(F)                    topology check chi = V-E+F, genus = (2-chi)/2
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ------------------------------------------------------------------ operator
def _tri_areas(V, F):
    u = V[F[:, 1]] - V[F[:, 0]]
    v = V[F[:, 2]] - V[F[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(u, v), axis=1)


def cotan_laplacian(V, F):
    """Cotangent-weighted stiffness W (PSD, discretizes -Delta_g) and lumped barycentric
    mass M (diagonal). The Laplace-Beltrami operator is M^{-1} W; -Delta eigenvalues solve
    the generalized problem W phi = lambda M phi."""
    n = len(V)
    i, j, k = F[:, 0], F[:, 1], F[:, 2]

    def cot_at(a, b, c):                       # cot of the angle at vertex a (a,b,c)
        u = V[b] - V[a]; v = V[c] - V[a]
        num = np.einsum("ij,ij->i", u, v)
        den = np.linalg.norm(np.cross(u, v), axis=1)
        return num / np.maximum(den, 1e-14)

    # angle at each vertex weights its OPPOSITE edge
    c_k = cot_at(k, i, j)                       # angle at k -> edge (i,j)
    c_i = cot_at(i, j, k)                       # angle at i -> edge (j,k)
    c_j = cot_at(j, k, i)                       # angle at j -> edge (k,i)
    I = np.concatenate([i, j, j, k, k, i])
    J = np.concatenate([j, i, k, j, i, k])
    w = 0.5 * np.concatenate([c_k, c_k, c_i, c_i, c_j, c_j])
    Woff = sp.coo_matrix((w, (I, J)), shape=(n, n)).tocsr()
    W = sp.diags(np.asarray(Woff.sum(axis=1)).ravel()) - Woff   # D - Woff  (PSD)

    area = _tri_areas(V, F)
    m = np.zeros(n)
    for c in range(3):                          # barycentric lumping: 1/3 area to each vertex
        np.add.at(m, F[:, c], area / 3.0)
    M = sp.diags(np.maximum(m, 1e-14))
    return W.tocsr(), M


def laplace_spectrum(V, F, k=12):
    """Smallest k eigenvalues (>=0) and eigenvectors of -Delta_g on the mesh."""
    W, M = cotan_laplacian(V, F)
    W = W.astype(np.float64).tocsc(); M = M.astype(np.float64).tocsc()
    vals, vecs = spla.eigsh(W, k=k, M=M, sigma=-1e-4, which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order], (W, M)


# ------------------------------------------------------------------ topology
def euler_genus(V, F):
    E = set()
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            E.add((u, v) if u < v else (v, u))
    chi = len(V) - len(E) + len(F)
    return chi, (2 - chi) // 2


# ------------------------------------------------------------------ generators
def icosphere(subdiv=3):
    """Unit sphere by recursively subdividing an icosahedron. Validation mesh."""
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(subdiv):
        mid = {}
        Vlist = list(V); newF = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = (V[a] + V[b]) / 2; p = p / np.linalg.norm(p)
                mid[key] = len(Vlist); Vlist.append(p)
            return mid[key]

        for a, b, c in F:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = np.array(Vlist); F = np.array(newF)
    return V, F


def two_tori_genus2(d=1.35, R=1.0, r=0.42, ngrid=110, pad=0.6):
    """Connected genus-2 surface = smooth union of two tori (rings in the xz-plane, hole
    axis y) whose tubes merge while both holes stay open. Meshed by marching cubes; genus
    is verified by euler_genus at the call site (tune d/R/r if not 2)."""
    from skimage.measure import marching_cubes

    def torus_field(x, y, z, cx):
        return (np.sqrt((x - cx) ** 2 + z ** 2) - R) ** 2 + y ** 2 - r ** 2

    ext = d + R + r + pad
    lin_xy = np.linspace(-ext, ext, ngrid)
    lin_z = np.linspace(-(R + r + pad), R + r + pad, max(40, ngrid // 2))
    X, Y, Z = np.meshgrid(lin_xy, np.linspace(-(r + pad), r + pad, max(30, ngrid // 3)),
                          lin_z, indexing="ij")
    vol = np.minimum(torus_field(X, Y, Z, -d), torus_field(X, Y, Z, +d))
    verts, faces, _, _ = marching_cubes(vol, level=0.0)
    # map voxel indices back to physical coordinates
    ax = [lin_xy, np.linspace(-(r + pad), r + pad, max(30, ngrid // 3)), lin_z]
    sp_ = [a[1] - a[0] for a in ax]; org = [a[0] for a in ax]
    V = np.column_stack([verts[:, c] * sp_[c] + org[c] for c in range(3)])
    return V, faces.astype(int)
