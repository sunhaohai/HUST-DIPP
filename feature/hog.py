#! -*-coding:utf8 -*-
import numpy as np

def rgb2gray(image):
    """彩色图变灰度图
    image.shape ==> [n, m, 3]
    return ==> [n, m]
    """
    r = np.squeeze(image[:, :, 0]) 
    g = np.squeeze(image[:, :, 1])
    b = np.squeeze(image[:, :, 2])
    gray = r*0.229 + g*0.587 + b*0.114
    return gray.astype('uint8')

def normalize(image):
    """图像归一化"""
    max = float(image.max())
    min = float(image.min())
    return ((image - min) / (max - min)) * 255.0

def adjust_gamma(image, gamma):
    """gamma矫正
    image.shape ==> [n, m]
    gamma ==> 指数幂
    return ==> [n, m]
    """
    result = (image / float(image.max())) ** gamma
    result = normalize(result)
    return result.astype('uint8')

def gradient(image):
    """计算梯度值
    image ==> 输入的图片 [n, m]
    return: 
      gx ==> x梯度 [n, m]
      gy ==> y梯度 [n, m]
    """
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[1:-1, :] = image[2:, :] - image[:-2, :]
    gy[:, 1:-1] = image[:, 2:] - image[:, :-2]
    return gx.astype('float'), gy.astype('float')

def cell_gradient(cell_grad, cell_angle, orientations):
    """计算cell直方图
    input:
      cell_grad: 梯度
      cell_angle: 角度
      orientations: 直方图表示等级
    output:
      orientation_centers: 直方图向量
    """
    orientation_centers = [0] * orientations
    angle_unit = 360 / orientations
    for i in range(cell_grad.shape[0]):
        for j in range(cell_grad.shape[1]):
            gradient_strength = cell_grad[i][j]
            gradient_angle = cell_angle[i][j]
            min_angle = int((gradient_angle / angle_unit) % orientations)
            max_angle = int((min_angle + 1) % orientations)
            mod = int(gradient_angle % angle_unit)
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / float(angle_unit))))
            orientation_centers[max_angle] += (gradient_strength * (mod / float(angle_unit)))
    return orientation_centers


def cell_gradient_vec(gx, gy, pix_per_cell=(8, 8), orientations=9):
    """计算cell向量
    input:
      gx: x方向的梯度
      gy: y方向的梯度
      pix_per_cell: 每个cell包含的像素点
      orientations: 直方图表示等级
    output:
      cell_gradient_vector: 直方图向量 [height/cell_size_x, width/cell_size_y, orientations]
    """
    height, width = gx.shape
    cell_gradient_vector = np.zeros([int(height/pix_per_cell[0]), int(width/pix_per_cell[1]), orientations])
    grand = (gx * gx + gy * gy) ** 0.5
    angle = np.arctan(gy/(gx + 99**-10)) * 180 * 4 / np.pi
    angle = np.nan_to_num(angle)

    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_grand = grand[i*pix_per_cell[0]:(i+1)*pix_per_cell[0],j*pix_per_cell[1]:(j+1)*pix_per_cell[1]]
            cell_angle = angle[i*pix_per_cell[0]:(i+1)*pix_per_cell[0],j*pix_per_cell[1]:(j+1)*pix_per_cell[1]]
            cell_gradient_vector[i][j] = cell_gradient(cell_grand, cell_angle, orientations)
    return cell_gradient_vector

def hog_block(cell_gradient_vector, block_size=(2, 2)):
    """计算hog向量大小
    input:
      cell_gradient_vector: 直方图向量
      block_size: block的大小
    output:
      hog_vector: hog特征
    """
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0]-block_size[0]+1):
        for j in range(cell_gradient_vector.shape[1]-block_size[1]+1):
            block_ = cell_gradient_vector[i:i+block_size[0], j:j+block_size[1], :].flatten()
            block_ = block_ / ((np.sum(block_*block_)**0.5)+10**-10)
            hog_vector.append(block_)
    return np.array(hog_vector)


def hog(image, cell=(8,8), block_size=(2,2), orientations=9):
    """hog算法
    input:
      image: 图片
      cell: cell大小
      block_size: block大小
      orientations: 直方图表示等级
    output:
      hog_vector: hog特征向量
    """
    if image.ndim != 2:
        image = rgb2gray(image)
    image = adjust_gamma(image, 0.5)
    gx, gy = gradient(image)
    cell_gradient_vector = cell_gradient_vec(gx, gy, cell, orientations)
    hog_vector = hog_block(cell_gradient_vector=cell_gradient_vector, block_size=block_size)
    return hog_vector.flatten()

if __name__ == "__main__":
    test = np.random.randint(0, 255, size=[32, 32 ,3])
    hog_vector = hog(test)
    print(hog_vector)
    #print(hog_vector.shape)
    



    