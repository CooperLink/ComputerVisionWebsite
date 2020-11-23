import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import io, util
import math
import cv2 

def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y+patchLength, x:x+overlap]
        error += np.sum(left**2)

    if y > 0:
        up   = patch[:overlap, :] - res[y:y+overlap, x:x+patchLength]
        error += np.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)

    return error

def minCutPath(errors):
    errors = np.pad(errors, [(0, 0), (1, 1)], 
                    mode='constant', 
                    constant_values=np.inf)

    cumError = errors[0].copy()
    paths = np.zeros_like(errors, dtype=int)    

    for i in range(1, len(errors)):
        M = cumError
        L = np.roll(M, 1)
        R = np.roll(M, -1)

        # optimize with np.choose?
        cumError = np.min((L, M, R), axis=0) + errors[i]
        paths[i] = np.argmin((L, M, R), axis=0)
    
    paths -= 1
    
    minCutPath = [np.argmin(cumError)]
    for i in reversed(range(1, len(errors))):
        minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]])
    
    return map(lambda x: x - 1, reversed(minCutPath))


def minCutPatch(patch, patchLength, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch

def bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    curPatchHeight, curPatchWidth = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            corrTexturePatch = corrTexture[i:i+curPatchHeight, j:j+curPatchWidth]
            e = corrTexturePatch - corrTargetPatch
            errors[i, j] = np.sum(e**2)

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+curPatchHeight, j:j+curPatchWidth]

def bestCorrOverlapPatch(texture, corrTexture, patchLength, overlap, 
                         corrTarget, res, y, x, alpha=0.1, level=0):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    di, dj = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+di, j:j+dj]
            l2error = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            overlapError = np.sum(l2error)

            corrTexturePatch = corrTexture[i:i+di, j:j+dj]
            corrError = np.sum((corrTexturePatch - corrTargetPatch)**2)

            prevError = 0
            if level > 0:
                prevError = patch[overlap:, overlap:] - res[y+overlap:y+patchLength, x+overlap:x+patchLength]
                prevError = np.sum(prevError**2)
            
            errors[i, j] = alpha * (overlapError + prevError) + (1 - alpha) * corrError

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+di, j:j+dj]

def transfer(texture, target, patchLength, mode="cut", 
             alpha=0.1, level=0, prior=None, blur=False):
    corrTexture = rgb2gray(texture)
    corrTarget  = rgb2gray(target)

    if blur:
        corrTexture = gaussian(corrTexture, sigma=3)
        corrTarget  = gaussian(corrTarget,  sigma=3)

    # remove alpha channel
    texture = util.img_as_float(texture)[:,:,:3]
    target = util.img_as_float(target)[:,:,:3]

    h, w, _ = target.shape
    overlap = patchLength // 6

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    
    if level == 0:
        res = np.zeros_like(target)
    else:
        res = prior
    print(numPatchesHigh)
    for i in range(numPatchesHigh):
        # print(i)
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "best":
                patch = bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x)
            elif mode == "overlap":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength, 
                                             overlap, corrTarget, res, y, x)
            elif mode == "cut":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength, 
                                             overlap, corrTarget, res, y, x, 
                                             alpha, level)
                patch = minCutPatch(patch, patchLength, overlap, res, y, x)
            
            res[y:y+patchLength, x:x+patchLength] = patch
      
    return res

def transferIter(texture, target, patchLength, n):
    res = transfer(texture, target, patchLength)
    for i in range(1, n):
        alpha = 0.1 + 0.8 * i / (n - 1)
        patchLength = patchLength * 2**i // 3**i
        print((alpha, patchLength))
        res = transfer(texture, target, patchLength, 
                       alpha=alpha, level=i, prior=res)
    
    return res

def implantImage(input, background):
    inputH, inputW, _ = input.shape
    totalH, totalW, _ = background.shape
    inputShape = np.array([inputH, inputW])
    backgroundShape = np.array([totalH, totalW])
    scale = max(1/4 * backgroundShape / inputShape)

    rescaled = cv2.resize(input, dsize=(int(scale * inputW), int(scale * inputH)), interpolation=cv2.INTER_CUBIC)
    rescaledH, rescaledW, _ = rescaled.shape

    newBackground = background.copy().astype(int)

    shiftH = 0
    shiftW = 0
    differences = np.Inf
    for sH in range(totalH - rescaledH):
        for sW in range(totalW - rescaledW):
            # dif = np.square(background[sH:sH + rescaledH, sW:sW + rescaledW] - rescaled[:,:])
            dif = (np.absolute(background[sH:sH + rescaledH, sW:sW + rescaledW] - rescaled[:,:]))
            if (np.sum(dif) < differences):
                shiftH = sH
                shiftW = sW
                differences = np.sum(dif)
    print("sH: ", shiftH)
    print("sW: ", shiftW)
    print(differences)
    newBackground[shiftH:shiftH + rescaledH, shiftW: shiftW + rescaledW] = (
        (1/2 * newBackground[shiftH:shiftH + rescaledH, shiftW: shiftW + rescaledW] + 1/2 * rescaled[:,:]))

    return newBackground.astype('uint8')

def textureTransfer(face, portrait):
    transferedInput = transferIter(portrait, face, 20, 2)
    implantedInput = implantImage(transferedInput, portrait)
    return implantedInput