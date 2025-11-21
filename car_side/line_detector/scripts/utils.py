#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def pixel_ratio(p1, p2, m):
    '''
    Calculate the ratio of pixels with value greater than 0.5 on the line segment between p1 and p2.

    Args:
        p1 (tuple): The starting point of the line segment (x1, y1).
        p2 (tuple): The ending point of the line segment (x2, y2).
        M (numpy.ndarray): The input image or matrix with pixel values.

    Returns:
        float: The ratio of pixels with value greater than 0.5 on the line segment.
    '''
    coords = linespace(p1, p2, m.shape)
    map_value = m[coords]

    ratio = ratio_seq(map_value)
    if len(coords[0]) == 0:
        return 0
    
    return ratio

def ratio_seq(seq):
    '''
    Calculate the ratio of values in the sequence that are greater than 0.5.

    Args:
        seq (numpy.ndarray): Input sequence of numerical values.

    Returns:
        float: The ratio of values greater than 0.5 to the total number of values.
    '''
    num = len(seq)
    nz_indexes = np.where(seq > 0.5)[0]
    if len(nz_indexes) == 0:
        return 0.0
    return len(nz_indexes) / num

def linespace(p1, p2, shape):
    '''
    Generate integer pixel coordinates along the line segment from p1 to p2, clipped to image bounds.

    Args:
        p1 (tuple): Start point (x1, y1).
        p2 (tuple): End point (x2, y2).
        shape (tuple): Image dimensions (height, width).

    Returns:
        tuple: Two aligned numpy arrays (y_coords, x_coords) ready for fancy indexing.
    '''
    x1, y1 = p1
    x2, y2 = p2
    h, w = shape

    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)

    num_x = max(x1, x2) - min(x1, x2) + 1
    num_y = max(y1, y2) - min(y1, y2) + 1

    if num_x < num_y:
        xlist = np.linspace(x1, x2, num=num_y)
        ylist = np.linspace(y1, y2, num=num_y)
    else:
        xlist = np.linspace(x1, x2, num=num_x)
        ylist = np.linspace(y1, y2, num=num_x)

    xlist = xlist.astype(np.int32)
    ylist = ylist.astype(np.int32)

    ylist[ylist > (h -1)] = h -1
    xlist[xlist > (w - 1)] = w - 1
    coords = np.vstack((ylist, xlist))
    return tuple(coords)

def pred_lines(image, displacement, scores, indices,
               score_thr=0.03,
               dist_thr=20.0):
    '''Post-processing process of the raw prediction'''

    _, w = image.shape
    expanded = cv2.dilate(image,np.ones((7,7)),iterations=1)
    valid_idx = np.where(scores > score_thr)
    scores = scores[valid_idx]
    indices = indices[valid_idx]

    yy = np.expand_dims(np.floor_divide(indices, w),-1)
    xx = np.expand_dims(np.fmod(indices, w),-1)
    pts = np.hstack((yy,xx))

    vmap = displacement.transpose((1,2,0))

    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, scores):
        y, x = center
        distance = dist_map[y, x]
        if distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end, distance, score])

    new_segments = np.array(segments_list)
    final_segments = []
    rat_lst = []
    for sgm in new_segments:
        sgm_st = sgm[:2]
        sgm_en = sgm[2:4]
        dist = sgm[4]
        rat = pixel_ratio(sgm_st, sgm_en, expanded)
        if rat > (1.1 - dist / (w * 1.5)):
            rat_lst.append((rat,dist))
            final_segments.append(sgm)
    final_segments = np.array(final_segments)
    if len(final_segments) == 0:
        return np.array([]), np.array([])

    '''post processing for decoupling'''
    # 1. get unique lines
    start = final_segments[:,:2]
    end = final_segments[:,2:4]
    scr = final_segments[:,-1]
    diff = start - end
    a = diff[:, 1]
    b = -diff[:, 0]
    c = a * start[:,0] + b * start[:,1]

    d = np.abs(c) / np.sqrt(a ** 2 + b ** 2 + 1e-10)
    theta = np.arctan2(diff[:,0], diff[:,1]) * 180 / np.pi
    theta[theta<0.0] += 180 #
    theta[theta>=180] -= 180

    # 2. hough and quantitize
    hough = np.concatenate([d[:,None], theta[:,None]], axis=-1)

    d_quant = 20
    theta_quant = 10

    hough[:,0] //= d_quant
    hough[:,1] //= theta_quant

    unique, indexs, inverse = np.unique(hough, axis=0, return_index=True, return_inverse=True)

    # 3. voting
    acc_map = np.zeros((2*w//d_quant, 180//theta_quant), dtype='float32')
    idx_map = np.zeros((2*w//d_quant, 180//theta_quant), dtype='int32') - 1
    unique = unique.astype('int32')
    for i, inv in enumerate(inverse):
        ind_x, ind_y = unique[inv][0], unique[inv][1]
        acc_map[ind_x, ind_y] += scr[i]
    idx_map[unique[:,0], unique[:,1]] = indexs
    acc_map_np = acc_map
 
    # 4. fast suppression using torch op
    acc_map = torch.from_numpy(acc_map_np).unsqueeze(0).unsqueeze(0)
    acc_map_get = torch.cat((acc_map[:,:,:,-2:],acc_map,acc_map[:,:,:,0:2]),dim=3)
    _,_, _, w = acc_map.shape
    max_acc_map = F.max_pool2d(acc_map_get,kernel_size=5, stride=1, padding=(2,0))
    acc_map = acc_map * ((acc_map == max_acc_map).float())
    flatten_acc_map = acc_map.reshape([-1, ])  
    indices = torch.nonzero(flatten_acc_map,as_tuple=False).reshape([-1,])
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    yx = torch.cat((yy, xx), dim=-1)
    yx = yx.detach().cpu().numpy()

    # 5. get segments
    merged_segments = []
    for yx_pt in yx:
        y_ori, x_ori = yx_pt
        indice = idx_map[y_ori, x_ori]
        index = np.where(indexs==indice)[0]
        indice_segments = (inverse == index)
        segment_list = final_segments[indice_segments][:,0:4].reshape([-1, 2])
        sorted_group_segments = np.sort(np.array(segment_list), axis=0)
        x_min, y_min = sorted_group_segments[0, :]
        x_max, y_max = sorted_group_segments[-1, :]
        deg = theta[indice]
        if (x_max - x_min) < 0.05 * (y_max - y_min):
            x_max = x_min = int((x_min+x_max)/2)
        elif (y_max - y_min) < 0.05 * (x_max - x_min):
            y_max = y_min = int((y_min + y_max)/2)
        dist = np.sqrt((x_max-x_min)**2 + (y_max-y_min)**2)
        if dist > 40:
            if deg >= 90:
                merged_segments.append([x_min, y_max, x_max, y_min])
            else:
                merged_segments.append([x_min, y_min, x_max, y_max])
    # 6. get intersections
    new_segments = np.array(merged_segments)

    if len(new_segments)==0:
        return new_segments, np.array([])
    start = new_segments[:, :2]
    end = new_segments[:, 2:]
    diff = start - end
    a = diff[:, 1]
    b = -diff[:, 0]
    c = a * start[:, 0] + b * start[:, 1]
    pre_det = a[:, None] * b[None, :]
    det = pre_det - np.transpose(pre_det)

    pre_inter_y = a[:, None] * c[None, :]
    inter_y = (pre_inter_y - np.transpose(pre_inter_y)) / (det + 1e-10)
    pre_inter_x = c[:, None] * b[None, :]
    inter_x = (pre_inter_x - np.transpose(pre_inter_x)) / (det + 1e-10)
    inter_pts = np.concatenate([inter_x[:, :, None], inter_y[:, :, None]], axis=-1).astype('int32').reshape([-1,2])
    inter_pts = np.unique(inter_pts,axis=0)[1:]
    return new_segments, inter_pts
