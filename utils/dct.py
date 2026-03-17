import numpy as np
from scipy.fftpack import dct, idct
# from .dct_for_loop import block_symmetric_quantize, block_symmetric_dequantize

def snr(original, reconstructed):
    """
    Compute the signal-to-noise ratio (in dB) between
    an original signal and its reconstruction.
    
    SNR(dB) = 10 * log10( sum(original^2) / sum((original - reconstructed)^2) )
    """
    noise = original - reconstructed
    signal_power = np.sum(original**2)
    noise_power = np.sum(noise**2)
    if noise_power < 1e-15:
        return 999.0  # effectively infinite if no difference
    return 10 * np.log10(signal_power / noise_power)


def inject_noise(x, noise_std=0.1):
    """
    Vectorized Gaussian noise injection per channel with normalization.

    Args:
        x (np.ndarray): Input of shape [C, H, W] or [B, C, H, W]
        noise_std (float): Scaling factor for noise relative to each channel std

    Returns:
        np.ndarray: Noisy array with same shape as input
    """
    # x = x  # ensure float ops, avoid double promotion

    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)  # [1, C, H, W]

    # Compute per-channel std over batch and spatial dimensions
    stds = np.std(x, axis=(0, 2, 3), keepdims=True)  # shape: [1, C, 1, 1]

    # Avoid division by zero
    stds = np.where(stds == 0, 1e-8, stds)

    # Generate Gaussian noise scaled per channel
    noise = np.random.randn(*x.shape) * stds * noise_std

    x_noisy = x + noise

    # return np.squeeze(x_noisy, axis=0) if x.shape[0] == 1 else x_noisy
    return x_noisy.astype(np.float32)


def dct2(block):
    """
    Compute 2D DCT of a 2D array `block`.
    Using type-II DCT in both dimensions (similar to what's used in JPEG).
    """
    # Apply DCT along one dimension
    temp = dct(block, axis=-2, norm='ortho')
    # Apply DCT along the other dimension
    dct_block = dct(temp, axis=-1, norm='ortho')
    return dct_block

def idct2(dct_block):
    """
    Compute inverse 2D DCT.
    """
    # Inverse DCT along one dimension
    temp = idct(dct_block, axis=-2, norm='ortho')
    # Inverse DCT along the other dimension
    recon_block = idct(temp, axis=-1, norm='ortho')
    return recon_block

def block_symmetric_quantize_vec(data, block_size, bits_out):

    # N, C, num_H, num_W, BL, _ = data.shape
    # Number of positive levels (excluding zero)
    # e.g., for 8 bits_out: levels = 2^(8-1) - 1 = 127
    levels = (1 << (bits_out - 1)) - 1  # 2^(bits_out-1) - 1
    # print(levels)
    
    # 1. Find max absolute value in this block
    block_maxabs = np.max(np.abs(data), axis=(-1,-2))
    # print(block_maxabs.shape)

    # 3. Compute quantization step
    step = block_maxabs / levels  # ==resolution e.g. M/127 if bits_out=8

    # 4. Quantize: q = round(x / step), clip to [-levels, levels]
    quantized_data = np.round(data / step[..., np.newaxis, np.newaxis]).astype(np.int32)
    quantized_data = np.clip(quantized_data, -levels, levels)

    return quantized_data, block_maxabs

def block_symmetric_dequantize_vec(quantized_data, block_maxabs, block_size, bits_out):

    levels = (1 << (bits_out - 1)) - 1  # 2^(bits_out-1) - 1

    step = block_maxabs / levels  # same step used in quantization
    reconstructed_data = quantized_data.astype(np.float32) * step[..., np.newaxis, np.newaxis]

    return reconstructed_data

def patchify(data, BL):
    """
    Convert a 4D tensor of shape (N, C, H, W) into blocks of size (BL x BL).
    Returns an array of shape (N, C, num_H, num_W, BL, BL).
    
    This can be done with np.reshape if H,W are multiples of BL, or
    more robustly with a small loop or stride_tricks.
    """
    N, C, H, W = data.shape
    num_H = H // BL
    num_W = W // BL
    # We'll assume H,W are exact multiples of BL for simplicity.
    
    # Reshape approach:
    out = data.reshape(N, C, num_H, BL, num_W, BL)  # -> (N, C, num_H, BL, num_W, BL)
    out = out.swapaxes(3,4)                         # -> (N, C, num_H, num_W, BL, BL)
    return out

def unpatchify(data, H=None, W=None):
    """
    Inverse of patchify.
    `data` is shape (N, C, num_H, num_W, BL, BL), 

    Returns shape (N, C, H, W).
    """
    if H is not None:
        if H != data.shape[2]*data.shape[4]:
            raise Exception("Specified H is incorrect")
    if W is not None:
        if W != data.shape[3]*data.shape[5]:
            raise Exception("Specified W is incorrect")
    N, C, num_H, num_W, BL, BL = data.shape
    H = int(num_H * BL)
    W = int(num_W * BL)
    # Undo the re-order
    out = data.swapaxes(3, 4)  # now shape (N, C, num_H, BL, num_W, BL)
    out = out.reshape(N, C, H, W)
    return out

def dct_based_compression(inputs, comp_ratio, BL, quantize=False, qbit=8, verify_quantize=False):

    dct_coef = np.zeros_like(inputs) # np.arr to store the DCT coefficients
    inputs_comp = np.zeros_like(inputs) # np.arr to store the compressed block

    num_H = int(inputs.shape[2]/BL) # horizontal pads
    num_W = int(inputs.shape[3]/BL) # vertical pads
    keep_ratio = 1/comp_ratio

    # (N, C, H, W) -> (N, C, num_H, num_W, BL, BL)
    inputs_patchify = patchify(inputs, BL)
    # print(f"inputs_patchify shape: {inputs_patchify.shape}") # (N, C, num_H, num_W, BL, BL)

    # DCT
    inputs_dct = dct2(inputs_patchify) 

    # Thresholding
    all_coeffs = np.abs(inputs_dct).reshape(inputs_dct.shape[:-2] + (-1,))  # (N, C, num_H, num_W, BL*BL)
    num_coeffs = all_coeffs.shape[-1]                                       # BL*BL
    num_to_keep = int(np.floor(keep_ratio * num_coeffs))                    # BL*BL*keep_ratio

    if num_to_keep <= 0:
        raise Exception("Keep ratio <= 0")
    elif num_to_keep > num_coeffs:
        raise Exception("Keep ratio > 1")
    else:
        sorted_coeffs_arr = np.flip(np.sort(all_coeffs, axis=-1), axis=-1)
        threshold_arr = sorted_coeffs_arr[:,:,:,:,num_to_keep - 1]
        inputs_dct[np.abs(inputs_dct) < threshold_arr[..., None, None]] = 0.0

    # Quantization + Dequant
    if quantize:
        # quantize the block
        q_dct, block_maxabs = block_symmetric_quantize_vec(inputs_dct, BL, qbit) 

        if verify_quantize:
            # print("check")
            # print(f"q_patch.max(): {q_patch.max()}")
            # print(f"q_patch.min(): {q_patch.min()} \n")
            if q_dct.min() < -2**(qbit-1) or q_dct.max() > 2**(qbit-1)-1:
                raise Exception("Quantized block out of range: [{}, {}]".format(q_dct.min(), q_dct.max()))

        # dequantize the block
        dq_dct = block_symmetric_dequantize_vec(q_dct, block_maxabs, BL, qbit)
        inputs_dct = dq_dct
    
    # IDCT
    inputs_comp_vec = idct2(inputs_dct) 
    inputs_comp_vec = unpatchify(inputs_comp_vec) # change back to 

    #########################################################
    # FOR DEBUGGING
    #########################################################
    '''
    for z in range(inputs.shape[1]):
        for x in range(num_H):
            for y in range(num_W):
                # print(np.sum(inputs_patchify[0,z,x,y] == inputs[0,z, x*BL:(x+1)*BL, y*BL:(y+1)*BL])) # check if before DCT are equal

                patch = dct2(inputs[0,z, x*BL:(x+1)*BL, y*BL:(y+1)*BL]) # apply DCT channelwise
                # print(np.sum(inputs_dct[0,z,x,y] == patch))

                all_coeffs = np.abs(patch).ravel()
                num_coeffs = all_coeffs.size
                num_to_keep = int(np.floor(keep_ratio * num_coeffs))
                # print(num_to_keep)

                # Edge case: if keep_ratio == 1.0, keep all coefficients
                if num_to_keep <= 0:
                    threshold = np.inf
                elif num_to_keep >= num_coeffs:
                    threshold = 0.0
                else:
                    sorted_coeffs = np.sort(all_coeffs)[::-1] # descending order
                    threshold = sorted_coeffs[num_to_keep - 1]
                    patch[np.abs(patch) < threshold] = 0.0 # zero-out small coefficients
                # print(np.array_equal(sorted_coeffs_arr[0,z,x,y],sorted_coeffs))
                # print(np.array_equal(threshold_arr[0,z,x,y],threshold))
                # print(np.array_equal(inputs_dct[0,z,x,y],patch))
                
                if quantize:
                    # quantize the block
                    q_patch, block_maxabs = block_symmetric_quantize(patch, BL, qbit)
                    # print(np.array_equal(q_dct[0,z,x,y],q_patch))

                    if verify_quantize:
                        # print("check")
                        # print(f"q_patch.max(): {q_patch.max()}")
                        # print(f"q_patch.min(): {q_patch.min()} \n")
                        if q_patch.min() < -2**(qbit-1) or q_patch.max() > 2**(qbit-1)-1:
                            raise Exception("Quantized block out of range: [{}, {}]".format(q_patch.min(), q_patch.max()))

                    # dequantize the block
                    dq_patch = block_symmetric_dequantize(q_patch, block_maxabs, BL, qbit)
                    patch = dq_patch
                    # print(np.array_equal(dq_dct[0,z,x,y],dq_patch))

                dct_coef[0,z, x*BL:(x+1)*BL, y*BL:(y+1)*BL] = patch
                inputs_comp[0,z, x*BL:(x+1)*BL, y*BL:(y+1)*BL] = idct2(patch)
    
    print(np.array_equal(inputs_comp_vec,inputs_comp))
    print(np.array_equal(unpatchify(inputs_dct),dct_coef))
    '''
    return inputs_comp_vec, unpatchify(inputs_dct)
    
