import numpy as np
import math


# for debugging
def get_single_target(ee_pos):
    point = np.array(list(ee_pos))
    point[1] -= 0.2
    point[0] += 0.2
    point[2] += 0.2
    return point


def eulidean_distance(cur, target):
    print("current: ", cur, "target: ", target)
    # convert tuple to np array
    cur = np.array(cur)
    return np.sqrt(np.sum(np.square(cur - target)))


def fibonacci_evenly_sampling_range_sphere(center, radius_arr, samples=100):
    '''
    Generate evenly distributed points for a range of radius value
    Let's consider a sphere with radius r1, r2, r3, ... rn
    We want to generate points on the sphere surface with radius r1, r2, r3, ... rn such that for every sphere space,
    the distance between points are the same.
    The number of points on the sphere surface is proportional to the area of the sphere surface, which is proportional to r^2

    :param center: coordinate [x, y, z] of the center of
    :param radius_arr:
    :param samples:
    :return:
    '''
    points = []
    # calculate no point allocations
    radius_arr = np.array(radius_arr)
    ratio_arr = np.square(radius_arr/radius_arr[0]) # ratio of no points = ratio of radius^2
    distribution_arr = ratio_arr/ np.sum(ratio_arr) # percentage of no points
    no_points_arr = np.round(distribution_arr * samples).astype(int)
    for idx, r in enumerate(radius_arr):
        no_points = no_points_arr[idx]
        # print("r: ", r,  "no_points: ", no_points)
        points.extend(fibonacci_evenly_sampling_sphere(center, r, no_points))
    return points


def fibonacci_evenly_sampling_sphere(center, radius, num_points=10):
    '''
    Reference: https://chiellini.github.io/2020/10/06/fibonacci_spiral_sphere/
    :param center:
    :param r:
    :param samples:
    :return:
    '''
    points = []
    phi = (np.pi / 2.) * (3. - np.sqrt(5.))  # golden angle in radians, but only in quarter sphere

    for i in range(num_points):
        lat = math.asin(-1.0 + 2.0 * float(i / (num_points + 1)))
        lon = phi * i

        x = math.cos(lon) * math.cos(lat) * radius + center[0]
        y = math.sin(lon) * math.cos(lat) * radius + center[1]
        z = math.sin(lat) * radius + center[2]

        # # Only keep the points that are in the first quadrant
        # if x >= center[0] and z >=center[2]:
        #     points.append([x, y, z])
        points.append([x, y, z])
    return points


def fibonacci_uniform_sampling_sphere(center, r, samples=100):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        # y = center[1] + r * (1 - (i / float(samples - 1)) * 2)  # y goes from 1 to -1
        # # radius = r * np.cbrt(np.random.uniform(0, 1))  # radius is random
        # radius = r * (i / float(samples - 1))
        sqrt_of_i_over_samples = np.sqrt(i / float(samples - 1))  # sqrt(i/samples) to ensure even distribution
        radius = r * sqrt_of_i_over_samples  # radius is proportional to sqrt(i)

        y = center[1] + r * (1 - (i / float(samples - 1)) * 2)  # y goes from 1 to -1
        theta = phi * i  # golden angle increment

        x = center[0] + np.cos(theta) * radius
        # y = center[1] + np.sin(theta) * radius
        z = center[2] + np.sin(theta) * radius

        points.append([x, y, z])

    return points


def uniform_sample(pos, radius, num_samples):
    """
    Sample points uniformly from the given space
    :param pos: (x, y, z)
    :return:
    """
    # pos = np.array(pos)
    # points = np.random.uniform(low=pos-radius, high=pos + radius, size=(num_samples, 3))
    points = []
    for i in range(num_samples):
        r = np.random.uniform(radius / 2, radius)
        theta = np.random.uniform(0, np.pi / 2)
        phi = np.random.uniform(0, np.pi / 2)  # Only sample from 0 to pi/2

        # Convert from spherical to cartesian coordinates
        dx = r * np.sin(phi) * np.cos(theta)
        dy = r * np.sin(phi) * np.sin(theta)
        dz = r * np.cos(phi)

        # Add to original point
        x_new = pos[0] + dx
        y_new = pos[1] + dy
        z_new = pos[2] + dz
        points.append([x_new, y_new, z_new])
    return points

