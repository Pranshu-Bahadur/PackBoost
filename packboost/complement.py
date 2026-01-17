import numpy as np

def float_to_packboost(X_float, resolution=3, use_quantiles=True):
    """
    Converts float features into multiple int8 features (values 0-4) using 
    hierarchical base-5 decomposition. This allows PackBoost to handle continuous
    variables with higher precision.
    
    Args:
        X_float: (N, F) float array.
        resolution: Number of integer components per original feature.
                    resolution=2 -> 25 bins (5^2)
                    resolution=3 -> 125 bins (5^3) [Recommended]
                    resolution=4 -> 625 bins (5^4) [High Precision]
        use_quantiles: If True, uses rank-based normalization (histogram equalization).
                       If False, uses min-max scaling.
    
    Returns:
        X_new: (N, F * resolution) int8 array ready for PackBoost.
    """
    if X_float.ndim != 2:
        raise ValueError("X_float must be 2D array")
        
    N, F = X_float.shape
    X_new = np.zeros((N, F * resolution), dtype=np.int8)
    
    for f in range(F):
        col = X_float[:, f]
        
        # 1. Normalize to [0, 0.99999]
        if use_quantiles:
            from scipy.stats import rankdata
            # Rank transform gives uniform distribution
            # 'dense' method ensures gaps are handled nicely
            ranks = rankdata(col, method='dense')
            col_norm = (ranks - 1) / np.max(ranks)
            col_norm = np.clip(col_norm, 0, 0.99999)
        else:
            # Min-Max scaling
            c_min, c_max = col.min(), col.max()
            if c_max > c_min:
                col_norm = (col - c_min) / (c_max - c_min)
                col_norm = np.clip(col_norm, 0, 0.99999)
            else:
                col_norm = np.zeros(N)

        # 2. Hierarchical Expansion (Base-5)
        current_val = col_norm
        for level in range(resolution):
            # Scale to 0-5 range
            scaled = current_val * 5.0
            
            # Extract integer digit (0, 1, 2, 3, or 4)
            digit = np.floor(scaled).astype(np.int8)
            
            # Place in new array:
            # Feature f occupies columns [f*resolution, f*resolution + resolution)
            X_new[:, f * resolution + level] = digit
            
            # Keep fractional part for the next level of precision
            current_val = scaled - digit

    return X_new


def int_to_packboost(X_int, max_val=None):
    """
    Converts integer features (high cardinality) into multiple int8 features 
    using exact Base-5 decomposition. Use this for counts, IDs, or categories.
    
    Args:
        X_int: (N, F) integer array (int32/int64/uint)
        max_val: Optional maximum value to determine fixed resolution.
                 If None, infers from data (per column or global).
                 
    Returns:
        X_new: (N, F * resolution) int8 array
    """
    if X_int.ndim != 2:
        raise ValueError("X_int must be 2D array")

    N, F = X_int.shape
    
    # 1. Determine required resolution (number of digits)
    if max_val is None:
        max_val = np.max(X_int)
    
    # How many base-5 digits needed? ceil(log5(max_val + 1))
    if max_val < 5:
        resolution = 1
    else:
        resolution = int(np.ceil(np.log(max_val + 1) / np.log(5)))
    
    print(f"Converting integers (max {max_val}) using {resolution} base-5 digits.")
    
    X_new = np.zeros((N, F * resolution), dtype=np.int8)
    
    # 2. Base-5 Decomposition (Little-Endian / LSD first)
    temp_X = X_int.astype(np.int64).copy() # Ensure int64 for safety
    
    for level in range(resolution):
        # Extract current digit (X % 5)
        digit = (temp_X % 5).astype(np.int8)
        
        # Store digit
        X_new[:, np.arange(F) * resolution + level] = digit
        
        # Integer divide for next level
        temp_X //= 5
        
    return X_new
