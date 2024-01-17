import pytest
import STASIS.Detector as df
import STASIS.Utility as util
import numpy as np
import math

def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors must have the same size")
    sum = 0
    for i in range(len(vec1)):
        sum += vec1[i]*vec2[i]

    return sum

def cross3d(vec1, vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors must have the same size")

    output = np.zeros(3)
    output[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    output[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    output[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0]

    return output

def norm(vec):

    sum = 0
    for i in range(len(vec)):
        sum += vec[i]**2

    return np.sqrt(sum)


def mult_quaternion(q1, q2):

    q_out = np.zeros(4)

    q_out[0] = q1[0]*q2[0] - dot(q1,q2)
    vec = q1[0]*q2[1:] + q2[0]*q1[1:] + cross3d(q1[1:], q2[1:])

    q_out[1] = vec[0]
    q_out[2] = vec[1]
    q_out[3] = vec[2]

    return q_out


def calc_quaternion(star, sc):

    quaternion_vector = cross3d(sc, star)

    quaternion = np.zeros(4)

    quaternion[0] = norm(sc)*norm(star)+dot(sc, star)
    quaternion[1] = quaternion_vector[0]
    quaternion[2] = quaternion_vector[1]
    quaternion[3] = quaternion_vector[2]

    quaternion = quaternion/norm(quaternion)

    return quaternion

def star_cat():
    stars = df.Stars("./test/testcatalouge.xml")
    return stars


def test_generators():
    poisson = util.rand_poisson_array(100, 100, 100)
    normal = util.rand_normal_array(100, 50, 100, 100)

    assert np.all(poisson)
    assert np.all(normal)

def test_resizing():
    base = np.ones((20,20))

    us = util.upsampling(base, 2, 0)
    ds = util.downsampling(base, 2)

    assert np.all(us)
    assert np.all(ds)
    assert math.isclose(np.sum(us), np.sum(base))
    assert math.isclose(np.sum(ds), np.sum(base))

def test_bias():
    bias = df.gen_bias(200, 20, 64, 64)
    assert np.all(bias)

def test_dark():
    hp = df.gen_hotpixels(0.01, 200, 100000, 64, 64)
    dark = df.gen_dark(120, 0.1, hp, 64, 64)
    assert np.all(dark)
    assert np.mean(hp) != 0

def test_flat():
    sb_flat = np.array([[1,1,1],[1,1,1],[1,1,1]])
    flat = df.gen_flat(0.97, 0.02, sb_flat, 0.92, 0.98, 0.002, 45, 1, 64, 64)

    assert np.all(flat)

def test_star_init():
    stars = star_cat()
    x, y, ra, dec, signal, is_target = stars.get_stars() 
    assert np.all(ra)
    assert np.all(dec)
    assert np.all(signal)
    assert len(x) == 2212

def test_star_shifts():
    stars = star_cat()
    

    stars.shift_stars(10, 10)
    x, y, ra, dec, signal, is_target = stars.get_stars()
    assert math.isclose(x[0], 10)
    assert math.isclose(y[0], 10)

    stars.rotate_star_position(180, 0, 0)
    x, y, ra, dec, signal, is_target = stars.get_stars()
    assert math.isclose(x[0], -10)
    assert math.isclose(y[0], -10)

def test_star_setters():
    stars = star_cat()
    stars.set_target_pos(10, 10)
    x,y,sig = stars.return_target_data()

    assert math.isclose(x, 10)
    assert math.isclose(y, 10)
    
    stars.update_star_signal(1000)
    x,y,sig = stars.return_target_data()
    assert math.isclose(sig, 267724)

def test_image_generation():
    stars = star_cat()
    
    ra = 292.84732636406
    dec = -10.89048807784

    star_vec = np.array((np.cos(dec*(np.pi/180))*np.cos(ra*(np.pi/180)), np.cos(dec*(np.pi/180))*np.sin(ra*(np.pi/180)), np.sin(dec*(np.pi/180))))
    sc_vec = np.array((0,0,1))
    quat = calc_quaternion(sc_vec, star_vec)

    psf = np.array([[0,1,0],[1,1,1],[0,1,0]])

    stars.convert_to_detector(quat, 1/360, 100)
 
    img = stars.gen_star_image(psf, 1., 1., 100, 100, 1, 5, 5)
    assert np.mean(img) != 0

    img_no_smear = stars.gen_star_image(psf, 1., 1., 100, 100, 1)

    assert np.mean(img_no_smear) != 0

    shot = df.gen_shotnoise(img)

    assert np.mean(shot) != 0 

def test_bkgr():
    psf = np.array([[0,1,0],[1,1,1],[0,1,0]])

    bkgr = df.gen_bkgr(psf, 10, 1, 1, 64, 64)

    assert np.all(bkgr)




