"""
ME144/244
LIDAR Simulation Module

Flash lidar simulation. A point emitter sends one pulse of rays over the camera field of view.
Rays increment in time and reflect off the surface. A ray that returns to the scanner height inside
the lens aperture is focused onto a pixel of the focal plane array, which records its arrival time
(time of flight). Each pixel's range is c*t/2 along that pixel's viewing direction.

FILL IN THE BLANK SECTIONS MARKED WITH TODO
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from matplotlib import cm

from lidar.surfaces import surfG, gradG, A_

###################################################################################################
############################### Simulation Constants and Parameters ###############################
###################################################################################################

c = 3e8                                         # speed of light (m/s)
z0 = 3                                          # scanner height (m)
nRays = 200000                                  # number of rays
dt = 0.001 * z0/c                               # time step size (s)
domainMax = 5                                   # sensor range: rays leaving |x|,|y| > domainMax are lost (m)
domainMin = -domainMax

# Flash lidar receiver (thin lens + focal plane array) and emitter
f = 0.1                                         # lens focal length (m)
focusDist = 1.0                                 # distance from the lens to the plane in focus (m)
sensorDist = f * focusDist / (focusDist - f)    # lens to focal plane array distance (thin lens eqn.)
pixelPitch = 4e-3                               # pixel size (m); with nPix this sets the field of view
nPix = 32                                       # pixels per side of the (square) focal plane array
apertureRadius = 0.1                            # lens aperture radius in the plane z = z0 (m)
diffuseFraction = 1.0                           # fraction of reflections that are diffuse (0 = mirror)
maxBounces = 10                                 # safety limit: rays that would reflect more times are lost
tanHalfFOV = nPix * pixelPitch / (2 * sensorDist)   # tan of the half field of view
emitter = np.array([0.0, 0.0, z0])              # point emitter, co-located with the lens center


error = 1e4 * np.ones(3)                        # initialize errors
rangeMap = {}                                   # range image of each surface (filled by lidarsim)
pointCloud = {}                                 # reconstructed point cloud of each surface
opticsUsed = {}                                 # lens settings of the last run of each surface


###################################################################################################
##################################### Simulation Functions ########################################
###################################################################################################

def lidarsim(surface, plot=True, focalLength=f):
    """
    Flash lidar simulation - a single diffuse pulse, a thin-lens receiver, and a focal plane array

    A point emitter at the center of the lens plane (0, 0, z0) sends a pulse of rays out over the
    camera field of view. Rays reflect off the surface (diffusely or specularly, possibly several
    times). A ray is DETECTED when it crosses the lens plane z = z0 inside the lens aperture. The lens focuses its
    arrival direction onto one pixel, and each pixel records only the arrival time of its first
    return. The receiver never knows where a ray was emitted or how many times it bounced.

    Parameters
    ----------
    surface : int
        Index into A_ array specifying which surface amplitude to use (0, 1, or 2)
    plot : bool, optional
        If True (default), plot the reconstructed point cloud at the end
    focalLength : float, optional
        Lens focal length (m) for this run. The pixel pitch and number of pixels stay fixed, so a
        longer focal length gives a narrower field of view and smaller pixel footprints.

    The results are stored in the module-level error, rangeMap, pointCloud and opticsUsed.
    """

    # Lens settings of THIS run (given): f, sensorDist and tanHalfFOV are local to this function
    f = focalLength
    sensorDist = f * focusDist / (focusDist - f)
    tanHalfFOV = nPix * pixelPitch / (2 * sensorDist)

    # FILL IN THE BLANK: Emit rays from the point emitter
    # The direction of each ray is given: random slopes tx, ty drawn uniformly over the field of
    # view, and the unit direction d0 = (tx, ty, -1) / |(tx, ty, -1)| pointing down at the surface.
    tx = (2 * np.random.rand(nRays) - 1) * tanHalfFOV
    ty = (2 * np.random.rand(nRays) - 1) * tanHalfFOV
    d0 = np.array([tx, ty, -np.ones(nRays)]).T
    d0 = d0 / np.linalg.norm(d0, axis=1)[:, np.newaxis]

    # TODO: All rays start at the emitter position (idealization) and move with speed c along d0
    # r = ...
    # v = ...

    # Tracking arrays
    active = np.ones(nRays, dtype=bool)
    rtrn = np.zeros(nRays, dtype=bool)
    rfln = np.zeros(nRays)
    tj = np.zeros(nRays)                        # time the ray crossed the lens plane
    uj = np.zeros((nRays, 3))                   # unit direction of the ray when detected
    rj = np.zeros((nRays, 3))                   # position of the ray when detected (on the lens)

    time = 0
    while any(active):

        # TODO: find indices of active rays at or below the surface
        # HINT: surfG() gives us surface height at x,y positions, compare this to the ray's height
        # idx = ...

        if idx.size != 0:
            # TODO: Compute the unit normal n (pointing down, into the surface) at the
            # intersection points.
            # HINT: the normal is (dG/dx, dG/dy, -1), normalized
            # n = ...

            # TODO: Compute v . n for these rays. A ray is only reflecting if it is moving INTO the
            # surface (v . n > 0).
            # vdotn = ...
            # entering = ...
            # idx, n, vdotn = idx[entering], n[entering], vdotn[entering]

            # Rays that have used up their allowed bounces are lost
            spent = rfln[idx] >= maxBounces
            active[idx[spent]] = False
            idx, n, vdotn = idx[~spent], n[~spent], vdotn[~spent]

            # Update reflection counter array
            rfln[idx] += 1

            # TODO: Specular reflection (law of reflection): v_ref = v - 2 (v . n) n  (n is a unit vector)
            # vSpec = ...

            # Diffuse (Lambertian) reflection: the ray leaves in a random direction about the
            # outward normal m = -n, with probability proportional to cos(angle from m)
            m = -n
            helper = np.where(np.abs(m[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
            t1 = np.cross(m, helper)
            t1 = t1 / np.linalg.norm(t1, axis=1)[:, np.newaxis]
            t2 = np.cross(m, t1)
            u1 = np.random.rand(len(idx))[:, np.newaxis]
            phi = 2 * np.pi * np.random.rand(len(idx))[:, np.newaxis]
            vDiff = c * (np.sqrt(u1) * np.cos(phi) * t1 + np.sqrt(u1) * np.sin(phi) * t2
                         + np.sqrt(1 - u1) * m)

            # Each reflection is diffuse with probability diffuseFraction, otherwise it is the
            # specular reflection vSpec that you computed above (given)
            isDiffuse = (np.random.rand(len(idx)) < diffuseFraction)[:, np.newaxis]
            v[idx, :] = np.where(isDiffuse, vDiff, vSpec)

        # TODO: Forward Euler time stepping - update ray positions
        # ONLY for active rays
        time += dt
        # r[...] += ...

        # Detection: an active ray that has crossed the lens plane going up (z > z0)
        # idx = ...
        # TODO: Of these, find the rays inside the lens aperture, a circle of apertureRadius^2
        # det = ...
        # rtrn[det] = True
        # TODO: Store the crossing time. The ray overshot the plane z = z0 by (z - z0) during the
        # last step, so remove the time spent overshooting the plane.
        # tj[det] = ...
        # rj[det, :] = ...
        # uj[det, :] = ...  (unit direction v / c)
        # TODO: Rays that crossed the plane are no longer active (inside the aperture or not)
        # active[idx] = ...

        # Rays that leave the sensor domain are lost (given)
        outside = np.logical_or(np.abs(r[:, 0]) > domainMax, np.abs(r[:, 1]) > domainMax)
        active[outside] = False

    # ---------------------------------------------------------------------------------------------
    # Thin-lens focal plane array (nothing to fill in here)
    #
    # The receiver is a camera: a thin lens with focal length f focuses the light that comes back
    # onto a focal plane array, a grid of pixels sitting a distance sensorDist behind the lens.
    # Each detected ray crosses the lens plane at the point h = (x, y) with a unit direction u.
    #
    #   1. The lens bends the ray. A ray with slope m = (u_x/u_z, u_y/u_z) that crosses the lens at
    #      a distance h from the axis leaves with slope m - h/f. (A ray parallel to the axis, m = 0,
    #      is bent toward the axis by h/f, so it crosses the axis one focal length f behind the lens.)
    #   2. The ray then travels sensorDist to the array, where it lands at p = h + sensorDist*slope.
    #      Light from a point at distance focusDist is brought to a single spot because sensorDist
    #      satisfies the thin-lens equation 1/f = 1/focusDist + 1/sensorDist. All rays that come from
    #      that point, whatever their h, land on the same pixel, so the pixel tells us the DIRECTION
    #      the light came from without knowing where in the lens it entered.
    #   3. The pixel index follows from where the ray lands (the array is centered on the axis).
    #   4. The time a pixel records is the time the light reached the lens center. A ray crossing
    #      the lens at h has already travelled a little farther from the object than a ray through
    #      the center (that is inside tj), and it travels farther to reach the array (path -
    #      sensorDist). An ideal lens exactly makes up for both by delaying rays LESS the farther
    #      they are from the axis (by |h|^2/(2f) less), so all rays from an in-focus point arrive at
    #      the same time. The tof line below adds the extra path and takes off that lens delay, which
    #      gives the time a ray through the lens center would have had.
    #
    # Light from points that are NOT at focusDist is not brought to a single spot, so it is spread
    # over several pixels (out of focus). 
    # ---------------------------------------------------------------------------------------------
    h = rj[rtrn, :2]                            # (x, y) where each detected ray crossed the lens
    u = uj[rtrn, :]                             # unit direction of each detected ray

    slope = u[:, :2] / u[:, 2:3] - h / f                # ray slope after the lens (step 1)
    pImg = h + sensorDist * slope                       # landing point on the array (step 2)
    col = np.floor(pImg[:, 0] / pixelPitch + nPix / 2).astype(int)      # pixel indices (step 3)
    row = np.floor(pImg[:, 1] / pixelPitch + nPix / 2).astype(int)
    onArray = (col >= 0) & (col < nPix) & (row >= 0) & (row < nPix)
    pix = row[onArray] * nPix + col[onArray]    # single pixel index of each ray that lands on the array

    # time registered by the pixel, referenced to the lens center (step 4)
    path = np.sqrt(np.sum((pImg - h) ** 2, axis=1) + sensorDist ** 2)
    tof = tj[rtrn] + (path - sensorDist - np.sum(h ** 2, axis=1) / (2 * f)) / c

    # Each pixel keeps only its FIRST return
    tPix = np.full(nPix * nPix, np.inf)
    np.minimum.at(tPix, pix, tof[onArray])
    hit = np.isfinite(tPix)

    # Viewing direction of each pixel center: the chief ray through the lens center
    cols, rows = np.meshgrid(np.arange(nPix), np.arange(nPix))
    xc = (cols.ravel() + 0.5 - nPix / 2) * pixelPitch
    yc = (rows.ravel() + 0.5 - nPix / 2) * pixelPitch
    dPix = np.array([-xc, -yc, -sensorDist * np.ones(nPix * nPix)]).T
    dPix = dPix / np.linalg.norm(dPix, axis=1)[:, np.newaxis]

    # TODO: Range of each pixel with a return from its round-trip time (emitter and receiver are
    # at the same place), then the 3D point along that pixel's viewing direction
    # R = ...
    # rp = ...

    # Store the range image (NaN = pixel with no return) for the comparison in reference.py (given)
    rangeMap[surface] = np.full(nPix * nPix, np.nan)
    rangeMap[surface][hit] = R
    pointCloud[surface] = rp
    opticsUsed[surface] = {'f': f, 'sensorDist': sensorDist, 'tanHalfFOV': tanHalfFOV}

    # TODO: Compute error (fraction of pixels with NO return)
    # error[surface] = ...

    # Point-cloud reconstruction plot
    if plot:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(rp[:, 0], rp[:, 1], rp[:, 2], marker='o', s=25, color='red', edgecolor='grey')
        ax.set_xlabel('X (m)', fontsize=15)
        ax.set_ylabel('Y (m)', fontsize=15)
        ax.set_zlabel('Z (m)', fontsize=15)
        ax.set_xlim((-0.75, 0.75))
        ax.set_ylim((-0.75, 0.75))
        plt.savefig('point_cloud_reconstruction_A{}.png'.format(A_[surface]))
        plt.show()

    print("Finished -> Error = ", error[surface])



def plot_surfaces():
    """
    Plot the three surface geometries before simulation
    """
    for i, _ in enumerate(A_):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-0.5, 0.5 + 0.01, 0.01)
        X, Y = np.meshgrid(x, y)
        zs = np.array(surfG(np.ravel(X), np.ravel(Y), A_[i]))
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z, cmap=cm.plasma)

        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
        ax.set_zlim(1.5, 4.0)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.savefig('surfacePlot_A{}.png'.format(i))
