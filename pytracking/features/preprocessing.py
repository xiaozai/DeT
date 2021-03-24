import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()


def sample_patch_transformed(im, pos, scale, image_sz, transforms, is_mask=False):
    """Extract transformed image samples.
    args:
        im: Image.
        pos: Center position for extraction.
        scale: Image scale to extract features from.
        image_sz: Size to resize the image samples to before extraction.
        transforms: A set of image transforms to apply.
    """

    # Get image patche
    im_patch, _ = sample_patch(im, pos, scale*image_sz, image_sz, is_mask=is_mask)

    # Apply transforms
    im_patches = torch.cat([T(im_patch, is_mask=is_mask) for T in transforms])

    return im_patches


def sample_patch_multiscale(im, pos, scales, image_sz, mode: str='replicate', max_scale_change=None):
    """Extract image patches at multiple scales.
    args:
        im: Image.
        pos: Center position for extraction.
        scales: Image scales to extract image patches from.
        image_sz: Size to resize the image samples to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """
    if isinstance(scales, (int, float)):
        scales = [scales]

    # Get image patches
    patch_iter, coord_iter = zip(*(sample_patch(im, pos, s*image_sz, image_sz, mode=mode,
                                                max_scale_change=max_scale_change) for s in scales))
    im_patches = torch.cat(list(patch_iter))
    patch_coords = torch.cat(list(coord_iter))

    return  im_patches, patch_coords


def sample_patch(im: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate', max_scale_change=None, is_mask=False):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.long().clone()

    pad_mode = mode

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        pad_mode = 'replicate'
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        shrink_factor = (sample_sz.float() / im_sz)
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        shrink_factor.clamp_(min=1, max=max_scale_change)
        sample_sz = (sample_sz.float() / shrink_factor).long()

    # Compute pre-downsampling factor
    if output_sz is not None:
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.float() / df     # new size

    # Do downsampling
    if df > 1:
        os = posl % df              # offset
        posl = (posl - os) // df     # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]   # downsample
    else:
        im2 = im

    # compute size to crop
    szl = torch.max(sz.round(), torch.Tensor([2])).long()

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl//2 + 1

    # Shift the crop to inside
    if mode == 'inside' or mode == 'inside_major':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        shift = (-tl - outside) * (outside > 0).long()
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

    # Get image patch
    if not is_mask:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), pad_mode)
    else:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))

    # Get image coordinates
    patch_coord = df * torch.cat((tl, br)).view(1,4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord

    # Resample
    if not is_mask:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
    else:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')

    return im_patch, patch_coord


def gaussian_prob(x, mu, std):
    '''
    Song : the depth value closer to the mu (center depth value) obtains the largest weights
    '''
    var = float(std)**2
    denom = (2*math.pi*var)**.5
    num = np.exp(-(x-float(mu))**2/(2*var))
    return num

def candidate_depth_from_histogram(depth_img, num_bins=8):
    '''
    Song :
    no proposals, so simply select teh
    '''

    # 1) crop the depth area
    dp_values = np.array(depth_img, dtype=np.float32).flatten()
    dp_values.sort()

    max_depth_img = np.max(depth_img.flatten())
    if len(dp_values) > 0:
        max_depth_box = dp_values[-1]
    else:
        prob_map = np.ones_like(depth_img, dtype=np.float32)
        candidate_depth = 0
        return prob_map, candidate_depth

    # 2) perform the historgram to count the frequency of the depth values
    hist, bin_edges = np.histogram(dp_values, bins=num_bins)
    hist = list(hist)

    hist_sorted = hist
    hist_sorted.sort()

    # 3) select the candidate depth
    '''
     select the second one as the candidate
     assume that the largest depth value belongs to background,
     # think, why we dont remove the background directly
    '''
    candidate_idx = hist.index(hist_sorted[-2]) #
    bin_lower = bin_edges[candidate_idx]
    bin_upper = bin_edges[candidate_idx+1]

    depth_in_bin = dp_values[dp_values>=bin_lower]
    depth_in_bin = depth_in_bin[depth_in_bin<=bin_upper]
    candidate_depth = np.mean(depth_in_bin) + 0.1 # to avoid the zeros

    # 4) obtain the depth-based Gaussian probability map
    img_sp = depth_img.shape
    sigma = min(img_sp[0], img_sp[1]) / 2.0
    prob_map = gaussian_prob(depth_img, candidate_depth, sigma)

    return prob_map, candidate_depth

def plot_prob_map(rgb_ori,  rgb_masked,  dp, prob_map):
    ''' Example Plot '''

    dp_colormap = cv.normalize(dp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    dp_colormap = cv.applyColorMap(dp_colormap, cv.COLORMAP_JET)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    ax1.imshow(rgb_ori)
    ax2.imshow(dp_colormap)
    ax3.imshow(rgb_masked)
    ax4.imshow(prob_map)

    plt.show()
    plt.close()

def sample_patch_hist_depth_mask(im: torch.Tensor, depth: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate', max_scale_change=None, is_mask=False, is_depth=False):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # print('song im shape : ', im.shape)    # (1, 3, 360, 640)
    # print('song dp shape : ', depth.shape) # (360, 640)

    # copy and convert
    posl = pos.long().clone()

    pad_mode = mode

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        pad_mode = 'replicate'
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        shrink_factor = (sample_sz.float() / im_sz)
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        shrink_factor.clamp_(min=1, max=max_scale_change)
        sample_sz = (sample_sz.float() / shrink_factor).long()

    # Compute pre-downsampling factor
    if output_sz is not None:
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.float() / df     # new size

    # Do downsampling
    if df > 1:
        os = posl % df              # offset
        posl = (posl - os) // df     # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]   # downsample
        dp2 = depth[os[0].item()::df, os[1].item()::df] # Song
    else:
        im2 = im
        dp2 = depth

    '''
    Song : performing depth mask here ?? it seems useless

          so performing nothing here during testing

          DO Nothing here !!!!!!!
    '''

    # compute size to crop
    szl = torch.max(sz.round(), torch.Tensor([2])).long()

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl//2 + 1

    # Shift the crop to inside
    if mode == 'inside' or mode == 'inside_major':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        shift = (-tl - outside) * (outside > 0).long()
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]
    # print(' song im2.shape : ', im2.shape) # (1, 3, 180, 320)
    # print(' song dp2.shape : ', dp2.shape) # (180, 320)

    # Get image patch
    if not is_mask:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), pad_mode)
    else:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))

    # print('song im_patch.shape : ', im_patch.shape) # (1, 3, 283, 283)
    # print('song dp_patch.shape : ', dp_path.shape)

    # Get image coordinates
    patch_coord = df * torch.cat((tl, br)).view(1,4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord

    # Resample
    if not is_mask:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
    else:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')

    return im_patch, patch_coord
