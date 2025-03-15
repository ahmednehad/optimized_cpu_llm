import numpy as np
import math
import time
import os
import json
import pickle
from threading import Thread
from multiprocessing import Pool, cpu_count
import ctypes
from functools import lru_cache
import numba
from numba import njit, prange, vectorize
import onnx
import onnxruntime as ort
import sentencepiece as spm

# Ensure memory alignment for SIMD operations
def align_memory(array, alignment=64):
    """Align array memory to the specified boundary for SIMD operations."""
       # If already aligned, return the original array
    if array.ctypes.data % alignment == 0:
        return array
            
    # Create a new array with the same shape but with aligned memory
    # We add extra padding to ensure we have enough space after alignment
    shape = array.shape
    size = array.size
    dtype = array.dtype
    itemsize = dtype.itemsize
    
    # Calculate how much extra space we need for alignment
    # Add alignment to ensure we have enough space
    extra_space = alignment // itemsize + 1
    
    # Create the buffer with extra padding
    buffer = np.empty(size + extra_space, dtype=dtype)
    
    # Find the offset needed to align the array
    offset = (alignment - buffer.ctypes.data % alignment) % alignment
    offset_elements = offset // itemsize
    
    # Create the aligned view
    aligned_view = buffer[offset_elements:offset_elements + size]
    
    # Reshape and copy data
    aligned_array = aligned_view.reshape(shape)
    aligned_array[:] = array
    print(f"Aligning array of shape {array.shape} with size {array.size}")
    print(f"Created aligned array of size {aligned_array.size}")

    return aligned_array
    

# SIMD-accelerated operations using Numba JIT compilation
@njit(fastmath=True, parallel=True)
def _matmul_numba_parallel(A, B, C):
    """
    Optimized matrix multiplication with parallelism and blocking.
    
    Args:
        A: First matrix (M x K)
        B: Second matrix (K x N)
        C: Output matrix (M x N)
    """
    M, K = A.shape
    N = B.shape[1]
    
    # Set block sizes based on CPU cache size (typically L1 cache)
    # Must be a constant for Numba to work with prange
    block_size = 64  # Typical L1 cache line size optimization
    
    # Compute in blocks for better cache utilization
    # Use constant step size of 1 for prange and handle block inside the loop
    for i_block in prange(0, (M + block_size - 1) // block_size):
        i_start = i_block * block_size
        i_end = min(i_start + block_size, M)
        
        for j_block in range(0, (N + block_size - 1) // block_size):
            j_start = j_block * block_size
            j_end = min(j_start + block_size, N)
            
            # Initialize block result to zero
            for ii in range(i_start, i_end):
                for jj in range(j_start, j_end):
                    C[ii, jj] = 0.0
            
            # Compute block result
            for k_block in range(0, (K + block_size - 1) // block_size):
                k_start = k_block * block_size
                k_end = min(k_start + block_size, K)
                
                for ii in range(i_start, i_end):
                    for kk in range(k_start, k_end):
                        a_val = A[ii, kk]
                        for jj in range(j_start, j_end):
                            C[ii, jj] += a_val * B[kk, jj]

@vectorize(["float32(float32)"], fastmath=True)
def _gelu_numba(x):
    """Vectorized GELU activation function."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

@njit(fastmath=True, parallel=True)
def _layer_norm_numba(x, weight, bias, eps):
    """
    Optimized layer normalization with parallelism.
    
    Args:
        x: Input tensor (batch_size, seq_len, hidden_size)
        weight: Scale parameter (hidden_size)
        bias: Bias parameter (hidden_size)
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor with same shape as x
    """
    batch_size, seq_len, hidden_size = x.shape
    result = np.empty_like(x)
    
    for b in prange(batch_size):
        for s in range(seq_len):
            # Compute mean and variance
            mean = 0.0
            for h in range(hidden_size):
                mean += x[b, s, h]
            mean /= hidden_size
            
            variance = 0.0
            for h in range(hidden_size):
                variance += (x[b, s, h] - mean) ** 2
            variance /= hidden_size
            
            # Normalize, scale and shift
            inv_std = 1.0 / np.sqrt(variance + eps)
            for h in range(hidden_size):
                result[b, s, h] = (x[b, s, h] - mean) * inv_std * weight[h] + bias[h]
                
    return result

@njit(fastmath=True)
def _softmax_numba(x):
    """
    Optimized softmax with improved numerical stability.
    
    Args:
        x: Input tensor (seq_len, vocab_size)
    
    Returns:
        Softmax probabilities
    """
    result = np.empty_like(x)
    
    # Find max for numerical stability
    for i in range(x.shape[0]):
        max_val = x[i, 0]
        for j in range(1, x.shape[1]):
            if x[i, j] > max_val:
                max_val = x[i, j]
        
        # Compute exp and sum
        exp_sum = 0.0
        for j in range(x.shape[1]):
            result[i, j] = np.exp(x[i, j] - max_val)
            exp_sum += result[i, j]
        
        # Normalize
        for j in range(x.shape[1]):
            result[i, j] /= exp_sum
            
    return result

# Define structured sparse attention patterns
class SparseAttentionPattern:
    """
    Implements efficient sparse attention patterns to reduce computation.
    """
    
    def __init__(self, max_seq_length, pattern_type='block_local_global'):
        """
        Initialize sparse attention pattern.
        
        Args:
            max_seq_length: Maximum sequence length
            pattern_type: Type of sparse pattern to use
                - 'block_local_global': Local window + strided global attention
                - 'local_window': Simple local window attention
                - 'strided': Strided attention pattern
                - 'full': Full attention (no sparsity)
        """
        self.max_seq_length = max_seq_length
        self.pattern_type = pattern_type
        self.pattern_cache = {}
        
    def get_sparse_mask(self, seq_length, local_window=128, global_stride=32):
        """
        Generate a sparse attention mask.
        
        Args:
            seq_length: Current sequence length
            local_window: Size of local attention window
            global_stride: Stride for global attention tokens
            
        Returns:
            Binary mask of shape (seq_length, seq_length) where 1 indicates attention
        """
        cache_key = (seq_length, local_window, global_stride)
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
            
        if self.pattern_type == 'full':
            # Full attention (no sparsity)
            mask = np.ones((seq_length, seq_length), dtype=np.float32)
            
        elif self.pattern_type == 'local_window':
            # Local window attention
            mask = np.zeros((seq_length, seq_length), dtype=np.float32)
            
            # Each token attends to tokens within a local window
            for i in range(seq_length):
                window_start = max(0, i - local_window // 2)
                window_end = min(seq_length, i + local_window // 2 + 1)
                mask[i, window_start:window_end] = 1.0
                
        elif self.pattern_type == 'strided':
            # Strided attention
            mask = np.zeros((seq_length, seq_length), dtype=np.float32)
            
            # Each token attends to tokens at regular intervals
            for i in range(seq_length):
                for j in range(0, seq_length, global_stride):
                    mask[i, j] = 1.0
                    
                # Always attend to self
                mask[i, i] = 1.0
                
        elif self.pattern_type == 'block_local_global':
            # Combined local window + strided global attention
            mask = np.zeros((seq_length, seq_length), dtype=np.float32)
            
            for i in range(seq_length):
                # Local window attention
                window_start = max(0, i - local_window // 2)
                window_end = min(seq_length, i + local_window // 2 + 1)
                mask[i, window_start:window_end] = 1.0
                
                # Strided global attention
                for j in range(0, seq_length, global_stride):
                    mask[i, j] = 1.0
                    
        else:
            raise ValueError(f"Unknown attention pattern: {self.pattern_type}")
            
        # Add causal mask (lower triangular)
        causal_mask = np.tril(np.ones((seq_length, seq_length), dtype=np.float32))
        mask = mask * causal_mask
        
        # Cache the mask
        self.pattern_cache[cache_key] = mask
        return mask

# Advanced quantization techniques
class AdvancedQuantization:
    """
    Implements advanced quantization techniques for model compression with improved
    support for embedding matrices and shape handling.
    """
    
    def __init__(self, quantize_type='int8', per_channel=True, optimize_zero_points=True):
        """
        Initialize quantization parameters.
        
        Args:
            quantize_type: Type of quantization ('int8', 'int4', 'mixed')
            per_channel: Whether to use per-channel quantization scales
            optimize_zero_points: Whether to optimize zero points
        """
        self.quantize_type = quantize_type
        self.per_channel = per_channel
        self.optimize_zero_points = optimize_zero_points
        
        # Set quantization parameters based on type
        if quantize_type == 'int8':
            self.num_bits = 8
            self.qmin = -128
            self.qmax = 127
        elif quantize_type == 'int4':
            self.num_bits = 4
            self.qmin = -8
            self.qmax = 7
        elif quantize_type == 'uint8':
            self.num_bits = 8
            self.qmin = 0
            self.qmax = 255
        elif quantize_type == 'mixed':
            # Mixed precision uses different bits for different layers
            self.num_bits = {'attention': 8, 'ffn': 8, 'embedding': 8}
            self.qmin = {'attention': -128, 'ffn': -128, 'embedding': -128}
            self.qmax = {'attention': 127, 'ffn': 127, 'embedding': 127}
        else:
            raise ValueError(f"Unsupported quantization type: {quantize_type}")
            
        self.scales = {}
        self.zero_points = {}
        self.original_shapes = {}  # Store original shapes for better handling
        
    def quantize_weight(self, weight, weight_name, layer_type='attention'):
        """
        Quantize a weight matrix with advanced techniques.
        
        Args:
            weight: Weight tensor to quantize
            weight_name: Name identifier for the weight
            layer_type: Type of layer ('attention', 'ffn', 'embedding')
            
        Returns:
            Quantized weight tensor
        """
        # Store original shape for reference
        self.original_shapes[weight_name] = weight.shape
        
        if self.quantize_type == 'mixed':
            num_bits = self.num_bits[layer_type]
            qmin = self.qmin[layer_type]
            qmax = self.qmax[layer_type]
        else:
            num_bits = self.num_bits
            qmin = self.qmin
            qmax = self.qmax
            
        # Determine quantization axis
        if self.per_channel:
            # For attention and FFN weights, quantize per output channel
            if len(weight.shape) == 2:
                axis = 0  # Quantize per output feature for linear layers
            else:
                axis = -1  # Last dimension for other weights
        else:
            axis = None  # Per-tensor quantization
            
        # Compute scale and zero point
        if axis is not None:
            # Per-channel quantization
            axes = tuple(i for i in range(len(weight.shape)) if i != axis)
            min_val = np.amin(weight, axis=axes, keepdims=True)
            max_val = np.amax(weight, axis=axes, keepdims=True)
        else:
            # Per-tensor quantization
            min_val = np.amin(weight)
            max_val = np.amax(weight)
            
        # Optimize zero point if enabled
        if self.optimize_zero_points:
            # Symmetric quantization centered around zero
            abs_max = np.maximum(np.abs(min_val), np.abs(max_val))
            scale = abs_max / ((qmax - qmin) / 2)
            zero_point = 0
        else:
            # Asymmetric quantization
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - np.round(min_val / scale) if scale != 0 else qmin
            
        # Handle division by zero
        scale = np.where(scale == 0, 1.0, scale)
            
        # Apply quantization
        quantized_weight = np.clip(np.round(weight / scale + zero_point), qmin, qmax).astype(np.int8)
        
        # Store scales and zero points
        self.scales[weight_name] = scale
        self.zero_points[weight_name] = zero_point
        
        return quantized_weight
        
    def dequantize_weight(self, quantized_weight, weight_name):
        """
        Dequantize a weight matrix or slice with improved shape handling.
        
        Args:
            quantized_weight: Quantized weight tensor or slice
            weight_name: Name identifier for the weight
            
        Returns:
            Dequantized weight tensor (floating point)
        """
        scale = self.scales[weight_name]
        zero_point = self.zero_points[weight_name]
        
        # Special handling based on tensor shapes
        original_shape = self.original_shapes.get(weight_name)
        
        # Check if this is a slice rather than the full tensor
        if original_shape and quantized_weight.shape != original_shape:
            # Special handling for embeddings - extract and dequantize individual vectors
            # This handles the token_embeddings[token_id] or position_embeddings[pos_id] case
            if len(quantized_weight.shape) == 1:
                # Single vector case (e.g., a single embedding vector)
                return (quantized_weight.astype(np.float32) - zero_point) * scale
                
            elif len(quantized_weight.shape) == 2 and quantized_weight.shape[0] == 1:
                # Sliced row with additional dimension (e.g., token_embeddings[token_id:token_id+1])
                # Return the vector without the extra dimension for direct assignment
                dequant = (quantized_weight.astype(np.float32) - zero_point) * scale
                return dequant[0]  # Remove the extra dimension
        
        # Standard dequantization for full tensors
        return (quantized_weight.astype(np.float32) - zero_point) * scale
        
    def quantize_activations(self, activation, name=None):
        """
        Quantize activation values (used for mixed-precision inference).
        
        Args:
            activation: Activation tensor
            name: Optional name for caching
            
        Returns:
            Quantized activation and scale factor
        """
        min_val = np.min(activation)
        max_val = np.max(activation)
        
        # Use full range of int8
        scale = (max_val - min_val) / 255
        zero_point = -128 - min_val / scale if scale != 0 else -128
        
        # Apply quantization
        quantized_activation = np.clip(np.round(activation / scale + zero_point), -128, 127).astype(np.int8)
        
        if name:
            self.scales[f"activation_{name}"] = scale
            self.zero_points[f"activation_{name}"] = zero_point
            
        return quantized_activation, scale, zero_point
        
    def dequantize_full_matrix(self, quantized_weight, weight_name):
        """
        Dequantize an entire weight matrix at once (more efficient).
        
        Args:
            quantized_weight: Full quantized weight tensor
            weight_name: Name identifier for the weight
            
        Returns:
            Dequantized weight tensor (floating point)
        """
        scale = self.scales[weight_name]
        zero_point = self.zero_points[weight_name]
        
        # Vectorized dequantization of the entire matrix
        return (quantized_weight.astype(np.float32) - zero_point) * scale
    
    def dequantize_single_row(self, quantized_weight, row_idx, weight_name):
        """
        Dequantize a single row/vector efficiently (for embedding lookup).
        
        Args:
            quantized_weight: Full quantized weight tensor
            row_idx: Index of the row to dequantize
            weight_name: Name identifier for the weight
            
        Returns:
            Dequantized vector (floating point)
        """
        scale = self.scales[weight_name]
        zero_point = self.zero_points[weight_name]
        
        # Extract the row and dequantize
        row = quantized_weight[row_idx]
        return (row.astype(np.float32) - zero_point) * scale
# Optimized KV cache with memory mapping
class OptimizedKVCache:
    """
    Memory-efficient key-value cache implementation using memory mapping.
    """
    
    def __init__(self, max_batch_size=1, max_seq_length=2048, num_layers=12, 
                 num_heads=12, head_dim=64, use_memory_mapping=True):
        """
        Initialize KV cache.
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_length: Maximum sequence length
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            use_memory_mapping: Whether to use memory-mapped files for very large caches
        """
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_memory_mapping = use_memory_mapping
        
        self.current_seq_length = 0
        self.cache = {}
        self.memmap_files = []
        
        # Create the cache
        self._initialize_cache()
        
    def _initialize_cache(self):
        """Initialize empty KV cache."""
        total_size_gb = (self.max_batch_size * self.max_seq_length * self.num_layers * 
                         self.num_heads * self.head_dim * 2 * 4) / (1024**3)
        
        print(f"Initializing KV cache with estimated size: {total_size_gb:.2f} GB")
        
        if self.use_memory_mapping and total_size_gb > 1.0:
            # Use memory mapping for large caches
            print("Using memory-mapped KV cache")
            self._initialize_memmap_cache()
        else:
            # Use in-memory cache for smaller models
            for i in range(self.num_layers):
                self.cache[i] = {
                    'key': np.zeros((self.max_batch_size, self.num_heads, 
                                    self.max_seq_length, self.head_dim), 
                                    dtype=np.float32),
                    'value': np.zeros((self.max_batch_size, self.num_heads, 
                                      self.max_seq_length, self.head_dim), 
                                      dtype=np.float32)
                }
                
    def _initialize_memmap_cache(self):
        """Initialize memory-mapped KV cache for large models."""
        # Create temporary directory for memmap files
        os.makedirs('kv_cache_tmp', exist_ok=True)
        
        # Create memory-mapped files for each layer
        for i in range(self.num_layers):
            # Create files for keys and values
            k_filename = f'kv_cache_tmp/layer_{i}_k.dat'
            v_filename = f'kv_cache_tmp/layer_{i}_v.dat'
            
            # Memory-map files
            k_memmap = np.memmap(k_filename, dtype=np.float32, mode='w+',
                               shape=(self.max_batch_size, self.num_heads, 
                                     self.max_seq_length, self.head_dim))
            v_memmap = np.memmap(v_filename, dtype=np.float32, mode='w+',
                               shape=(self.max_batch_size, self.num_heads, 
                                     self.max_seq_length, self.head_dim))
                                     
            # Initialize to zeros
            k_memmap[:] = 0
            v_memmap[:] = 0
            
            # Add to cache
            self.cache[i] = {
                'key': k_memmap,
                'value': v_memmap
            }
            
            # Keep track of files for cleanup
            self.memmap_files.extend([k_filename, v_filename])
            
    def update(self, layer_idx, new_keys, new_values):
        """
        Update KV cache with new keys and values.
        
        Args:
            layer_idx: Layer index
            new_keys: New key states to add [batch_size, num_heads, seq_len, head_dim]
            new_values: New value states to add [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, _, new_seq_len, _ = new_keys.shape
        
        # For the first token, reset sequence length
        if self.current_seq_length == 0 or new_seq_len > 1:
            self.current_seq_length = new_seq_len
        else:
            # For subsequent tokens, append to cache
            old_seq_len = self.current_seq_length
            self.current_seq_length += 1
            
            # Ensure we don't exceed maximum sequence length
            if self.current_seq_length > self.max_seq_length:
                # Shift KV cache to make room by removing oldest tokens
                shift_amount = self.current_seq_length - self.max_seq_length
                self.cache[layer_idx]['key'][:, :, :-shift_amount, :] = self.cache[layer_idx]['key'][:, :, shift_amount:, :]
                self.cache[layer_idx]['value'][:, :, :-shift_amount, :] = self.cache[layer_idx]['value'][:, :, shift_amount:, :]
                self.current_seq_length = self.max_seq_length
                
            # Add new keys and values
            self.cache[layer_idx]['key'][:batch_size, :, old_seq_len:self.current_seq_length, :] = new_keys
            self.cache[layer_idx]['value'][:batch_size, :, old_seq_len:self.current_seq_length, :] = new_values
            
    def get(self, layer_idx, batch_size=None):
        """
        Get cached keys and values for a layer.
        
        Args:
            layer_idx: Layer index
            batch_size: Batch size to retrieve (defaults to all batches)
            
        Returns:
            Tuple of (keys, values) for the specified layer
        """
        if batch_size is None:
            batch_size = self.max_batch_size
            
        if self.current_seq_length == 0:
            return None, None
            
        # Get current keys and values
        keys = self.cache[layer_idx]['key'][:batch_size, :, :self.current_seq_length, :]
        values = self.cache[layer_idx]['value'][:batch_size, :, :self.current_seq_length, :]
        
        return keys, values
        
    def clear(self):
        """Clear the KV cache."""
        self.current_seq_length = 0
        
        # Re-initialize cache to zeros
        for i in range(self.num_layers):
            if i in self.cache:
                self.cache[i]['key'].fill(0)
                self.cache[i]['value'].fill(0)
                
    def __del__(self):
        """Clean up memory-mapped files."""
        # Close memory-mapped files
        for i in range(self.num_layers):
            if i in self.cache:
                if hasattr(self.cache[i]['key'], '_mmap'):
                    self.cache[i]['key']._mmap.close()
                if hasattr(self.cache[i]['value'], '_mmap'):
                    self.cache[i]['value']._mmap.close()
                    
        # Remove temporary files
        for filename in self.memmap_files:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
                    
        # Remove directory if empty
        try:
            if os.path.exists('kv_cache_tmp') and len(os.listdir('kv_cache_tmp')) == 0:
                os.rmdir('kv_cache_tmp')
        except:
            pass

# Optimized tokenizer integration
class OptimizedTokenizer:
    """
    Optimized tokenizer implementation with fallback strategies.
    """
    
    def __init__(self, vocab_size=32000, model_path=None):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
            model_path: Path to SentencePiece model file
        """
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.sp_model = None
        
        # Special tokens
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        
        # Initialize tokenizer
        self._initialize_tokenizer()
        
    def _initialize_tokenizer(self):
        """Initialize the appropriate tokenizer."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Use SentencePiece if model file exists
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.Load(self.model_path)
                print(f"Loaded SentencePiece model from {self.model_path}")
                return
            except Exception as e:
                print(f"Error loading SentencePiece model: {e}")
                
        # Fallback to character-level tokenization
        print("Using fallback character-level tokenizer")
        
    def encode(self, text, add_special_tokens=True):
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        if self.sp_model:
            # Use SentencePiece
            tokens = self.sp_model.EncodeAsIds(text)
        else:
            # Character-level fallback
            tokens = [ord(c) % (self.vocab_size - 100) + 100 for c in text]
            
        # Add special tokens if requested
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
            
        return np.array(tokens, dtype=np.int32)
        
    def decode(self, token_ids):
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Remove special tokens
        token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
        
        if self.sp_model:
            # Use SentencePiece
            text = self.sp_model.DecodeIds(token_ids)
        else:
            # Character-level fallback
            text = ''.join([chr((t - 100) % 26 + ord('a')) if t >= 100 else '?' for t in token_ids])
            
        return text
        
    def batch_encode(self, texts, max_length=None, padding=True, truncation=True):
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences to the same length
            truncation: Whether to truncate sequences longer than max_length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        input_ids = []
        
        for text in texts:
            tokens = self.encode(text)
            
            # Truncate if needed
            if truncation and max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
                
            input_ids.append(tokens)
            
        # Determine max length in batch if padding
        if padding:
            if max_length:
                pad_len = max_length
            else:
                pad_len = max(len(ids) for ids in input_ids)
                
            # Create attention mask and pad sequences
            attention_mask = []
            
            for i, ids in enumerate(input_ids):
                pad_count = pad_len - len(ids)
                attention_mask.append(np.concatenate([
                    np.ones(len(ids), dtype=np.int32),
                    np.zeros(pad_count, dtype=np.int32)
                ]))
                
                # Pad input IDs
                input_ids[i] = np.concatenate([
                    ids,
                    np.full(pad_count, self.pad_token_id, dtype=np.int32)
                ])
                
            # Convert to arrays
            input_ids = np.array(input_ids, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.int32)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            # No padding, just return input IDs
            return {
                'input_ids': input_ids
            }

# Main optimized LLM implementation
class OptimizedCPULLM:
    """
    Fully optimized LLM implementation for CPU inference with advanced optimizations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the optimized CPU LLM.
        
        Args:
            config: Dictionary with model configuration
        """
        # Default configuration
        self.default_config = {
            "vocab_size": 32000,
            "hidden_size": 768,
            "n_layers": 8,
            "n_heads": 12,
            "head_size": 64,
            "intermediate_size": 2048,
            "max_seq_length": 1024,
            "activation": "gelu",
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
            "use_bias": True,
            
            # Optimization settings
            "quantize": True,
            "quantize_type": "int8",
            "per_channel_quantization": True,
            "optimize_zero_points": True,
            "sparse_attention": True,
            "sparse_attention_pattern": "block_local_global",
            "local_attention_window": 128,
            "global_attention_stride": 32,
            "use_memory_mapping": True,
            "use_numba": True,
            "use_onnx": False,
            "use_kv_cache": True,
            "use_threading": True,
            "use_numa_aware": True,
            "num_threads": None,  # Auto-detect
            "block_size": 64,
            "profile_execution": False
        }
        
        # Update with user config if provided
        if config:
            self.config = {**self.default_config, **config}
        else:
            self.config = self.default_config
            
        # Get number of CPU cores
        self.num_cpus = cpu_count()
        if self.config["num_threads"] is None:
            self.config["num_threads"] = self.num_cpus
            
        # Extract config values
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]
        self.n_layers = self.config["n_layers"]
        self.n_heads = self.config["n_heads"]
        self.head_size = self.config["head_size"]
        self.intermediate_size = self.config["intermediate_size"]
        self.max_seq_length = self.config["max_seq_length"]
        
        # Initialize model components
        self.token_embeddings = None
        self.position_embeddings = None
        self.layers = []
        self.ln_f = None
        self.lm_head = None
        
        # Initialize optimization components
        if self.config["quantize"]:
            self.quantizer = AdvancedQuantization(
                quantize_type=self.config["quantize_type"],
                per_channel=self.config["per_channel_quantization"],
                optimize_zero_points=self.config["optimize_zero_points"]
            )
        else:
            self.quantizer = None
            
        if self.config["sparse_attention"]:
            self.sparse_attention = SparseAttentionPattern(
                max_seq_length=self.max_seq_length,
                pattern_type=self.config["sparse_attention_pattern"]
            )
        else:
            self.sparse_attention = None
            
        if self.config["use_kv_cache"]:
            self.kv_cache = OptimizedKVCache(
                max_batch_size=1,  # Default for inference
                max_seq_length=self.max_seq_length,
                num_layers=self.n_layers,
                num_heads=self.n_heads,
                head_dim=self.head_size,
                use_memory_mapping=self.config["use_memory_mapping"]
            )
        else:
            self.kv_cache = None
            
        # Initialize tokenizer
        self.tokenizer = None
        
        # ONNX model
        self.onnx_model = None
        self.onnx_session = None
        
        # Performance tracking
        self.profiling = {}
        
        # Memory usage tracking
        self.memory_used = 0
        
        print(f"Initialized optimized CPU LLM with {self.get_parameter_count()/1e6:.2f}M parameters")
        
    def initialize_model(self, seed=42):
        """
        Initialize model weights.
        
        Args:
            seed: Random seed for initialization
        """
        np.random.seed(seed)
        print(f"Initializing model with random seed {seed}")
        
        # Initialize token embeddings
        self.token_embeddings = np.random.randn(
            self.vocab_size, self.hidden_size
        ).astype(np.float32) * self.config["initializer_range"]
        self.token_embeddings = align_memory(self.token_embeddings)
        
        # Initialize position embeddings
        self.position_embeddings = np.random.randn(
            self.max_seq_length, self.hidden_size
        ).astype(np.float32) * self.config["initializer_range"]
        self.position_embeddings = align_memory(self.position_embeddings)
        
        # Initialize layers
        for i in range(self.n_layers):
            layer = {}
            
            # Attention weights
            for weight_name in ['query', 'key', 'value']:
                layer[weight_name] = np.random.randn(
                    self.hidden_size, self.n_heads * self.head_size
                ).astype(np.float32) * self.config["initializer_range"]
                layer[weight_name] = align_memory(layer[weight_name])
                
            layer['attn_output'] = np.random.randn(
                self.n_heads * self.head_size, self.hidden_size
            ).astype(np.float32) * self.config["initializer_range"]
            layer['attn_output'] = align_memory(layer['attn_output'])
            
            # Attention biases if enabled
            if self.config["use_bias"]:
                for bias_name in ['query_bias', 'key_bias', 'value_bias', 'attn_output_bias']:
                    layer[bias_name] = np.zeros(
                        self.n_heads * self.head_size if bias_name != 'attn_output_bias' else self.hidden_size
                    ).astype(np.float32)
                    layer[bias_name] = align_memory(layer[bias_name])
            
            # Layer norms
            for ln_name in ['ln_1', 'ln_2']:
                layer[f'{ln_name}_weight'] = np.ones(self.hidden_size).astype(np.float32)
                layer[f'{ln_name}_bias'] = np.zeros(self.hidden_size).astype(np.float32)
                layer[f'{ln_name}_weight'] = align_memory(layer[f'{ln_name}_weight'])
                layer[f'{ln_name}_bias'] = align_memory(layer[f'{ln_name}_bias'])
            
            # FFN weights
            layer['fc_1'] = np.random.randn(
                self.hidden_size, self.intermediate_size
            ).astype(np.float32) * self.config["initializer_range"]
            layer['fc_2'] = np.random.randn(
                self.intermediate_size, self.hidden_size
            ).astype(np.float32) * self.config["initializer_range"]
            layer['fc_1'] = align_memory(layer['fc_1'])
            layer['fc_2'] = align_memory(layer['fc_2'])
            
            # FFN biases if enabled
            if self.config["use_bias"]:
                layer['fc_1_bias'] = np.zeros(self.intermediate_size).astype(np.float32)
                layer['fc_2_bias'] = np.zeros(self.hidden_size).astype(np.float32)
                layer['fc_1_bias'] = align_memory(layer['fc_1_bias'])
                layer['fc_2_bias'] = align_memory(layer['fc_2_bias'])
            
            self.layers.append(layer)
            
        # Final layer norm
        self.ln_f = {
            'weight': np.ones(self.hidden_size).astype(np.float32),
            'bias': np.zeros(self.hidden_size).astype(np.float32)
        }
        self.ln_f['weight'] = align_memory(self.ln_f['weight'])
        self.ln_f['bias'] = align_memory(self.ln_f['bias'])
        
        # LM head (tied with token embeddings)
        self.lm_head = self.token_embeddings
        
        # Initialize tokenizer
        self.tokenizer = OptimizedTokenizer(vocab_size=self.vocab_size)
        
        # Calculate memory usage
        self.calculate_memory_usage()
        
        # Apply quantization if configured
        if self.config["quantize"]:
            self.quantize_model()
            
        print(f"Model initialized with estimated memory usage: {self.memory_used / (1024**2):.2f} MB")
        
    def calculate_memory_usage(self):
        """Calculate model memory usage."""
        self.memory_used = 0
        
        # Embeddings
        self.memory_used += self.token_embeddings.nbytes
        self.memory_used += self.position_embeddings.nbytes
        
        # Layers
        for layer in self.layers:
            for param in layer.values():
                self.memory_used += param.nbytes
                
        # Final layer norm
        self.memory_used += self.ln_f['weight'].nbytes + self.ln_f['bias'].nbytes
        
        return self.memory_used
        
    def get_parameter_count(self):
        """Calculate number of parameters in the model."""
        count = 0
        
        # Embeddings
        count += self.vocab_size * self.hidden_size  # Token embeddings
        count += self.max_seq_length * self.hidden_size  # Position embeddings
        
        # Per layer
        for _ in range(self.n_layers):
            # Self-attention
            count += 3 * self.hidden_size * self.n_heads * self.head_size  # QKV projections
            count += self.n_heads * self.head_size * self.hidden_size  # Output projection
            
            # Biases if enabled
            if self.config["use_bias"]:
                count += 3 * self.n_heads * self.head_size  # QKV biases
                count += self.hidden_size  # Output projection bias
            
            # Layer norms
            count += 2 * 2 * self.hidden_size  # Weights and biases for 2 layer norms
            
            # FFN
            count += self.hidden_size * self.intermediate_size  # FC1
            count += self.intermediate_size * self.hidden_size  # FC2
            
            # FFN biases if enabled
            if self.config["use_bias"]:
                count += self.intermediate_size  # FC1 bias
                count += self.hidden_size  # FC2 bias
        
        # Final layer norm
        count += 2 * self.hidden_size
        
        return count
        
    def quantize_model(self):
        """Quantize model weights."""
        print(f"Quantizing model with {self.config['quantize_type']} precision")
        
        # Quantize embeddings
        self.token_embeddings = self.quantizer.quantize_weight(
            self.token_embeddings, 'token_embeddings', 'embedding'
        )
        
        self.position_embeddings = self.quantizer.quantize_weight(
            self.position_embeddings, 'position_embeddings', 'embedding'
        )
        
        # Quantize layers
        for i, layer in enumerate(self.layers):
            # Attention weights
            for weight_name in ['query', 'key', 'value', 'attn_output']:
                if weight_name in layer:
                    layer[weight_name] = self.quantizer.quantize_weight(
                        layer[weight_name], f'layer_{i}_{weight_name}', 'attention'
                    )
            
            # FFN weights
            for weight_name in ['fc_1', 'fc_2']:
                if weight_name in layer:
                    layer[weight_name] = self.quantizer.quantize_weight(
                        layer[weight_name], f'layer_{i}_{weight_name}', 'ffn'
                    )
        
        # Recalculate memory usage
        self.calculate_memory_usage()
        print(f"Model quantized. New memory usage: {self.memory_used / (1024**2):.2f} MB")
        
    def forward(self, input_ids, attention_mask=None, use_kv_cache=None, profile=None):
        """
        Forward pass through the model with optimizations.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            use_kv_cache: Whether to use KV caching
            profile: Whether to profile execution time
            
        Returns:
            Logits for next token prediction
        """
        if profile is None:
            profile = self.config["profile_execution"]
            
        # Use kv_cache if configured
        if use_kv_cache is None:
            use_kv_cache = self.config["use_kv_cache"] and self.kv_cache is not None
            
        # Start profiling if enabled
        if profile:
            profiling_data = {}
            start_time = time.time()
            
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        if profile:
            embed_start = time.time()
            
        # Process token embeddings
        if self.config["quantize"]:
            # Dequantize entire embedding matrices first - this is more efficient
            # Get token embedding scales and zero points
            token_scale = self.quantizer.scales['token_embeddings']
            token_zero_point = self.quantizer.zero_points['token_embeddings']
            
            # Get position embedding scales and zero points
            pos_scale = self.quantizer.scales['position_embeddings']
            pos_zero_point = self.quantizer.zero_points['position_embeddings']
            
            # Prepare hidden states
            hidden_states = np.zeros((batch_size, seq_length, self.hidden_size), dtype=np.float32)
            
            # If we're in generation mode with a single token, optimize
            if input_ids.shape[1] == 1 and self.kv_cache and self.kv_cache.current_seq_length > 0:
                # Process a single token
                for b in range(batch_size):
                    for s in range(seq_length):
                        # Get token embedding
                        token_id = input_ids[b, s]
                        if token_id < self.vocab_size:
                           # Ensure we're getting a single vector by explicitly indexing
                            if isinstance(self.token_embeddings, np.ndarray) and len(self.token_embeddings.shape) == 2:
                                token_embedding = self.token_embeddings[token_id, :]  # Explicit indexing to ensure we get a vector
                            else:
                                # Fallback to using the quantizer's dequantize_single_row method
                                hidden_states[b, s] = self.quantizer.dequantize_single_row(
                                    self.token_embeddings, token_id, 'token_embeddings'
                                )
                                continue  # Skip the rest of this iteration
                        # Get position embedding
                        pos_id = seq_length - 1 + self.kv_cache.current_seq_length
                        if pos_id < self.max_seq_length:
                            # Fix: Make sure to get just the embedding vector for this position
                            pos_embedding = self.position_embeddings[pos_id].copy()  # Get single vector
                            # Dequantize position embedding
                            pos_embedding_dequant = (pos_embedding.astype(np.float32) - pos_zero_point) * pos_scale
                            # Add to hidden states
                            hidden_states[b, s] += pos_embedding_dequant
            else:
                # Process full sequence - dequantize entire matrices for efficiency
                # Dequantize token embeddings
                dequantized_token_emb = (self.token_embeddings.astype(np.float32) - token_zero_point) * token_scale
                
                # Dequantize position embeddings
                dequantized_pos_emb = (self.position_embeddings.astype(np.float32) - pos_zero_point) * pos_scale
                
                # Fill hidden states with token embeddings
                for b in range(batch_size):
                    for s in range(seq_length):
                        token_id = input_ids[b, s]
                        if token_id < self.vocab_size:
                            hidden_states[b, s] = dequantized_token_emb[token_id]
                
                # Add position embeddings
                position_ids = np.arange(seq_length)[None, :].repeat(batch_size, axis=0)
                for b in range(batch_size):
                    for s in range(seq_length):
                        pos_id = position_ids[b, s]
                        if pos_id < self.max_seq_length:
                            hidden_states[b, s] += dequantized_pos_emb[pos_id]
                            
        else:
            # Non-quantized version - uses standard embedding lookup
            hidden_states = np.zeros((batch_size, seq_length, self.hidden_size), dtype=np.float32)
            
            # Add token embeddings
            for b in range(batch_size):
                for s in range(seq_length):
                    token_id = input_ids[b, s]
                    if token_id < self.vocab_size:
                        hidden_states[b, s] = self.token_embeddings[token_id]
            
            # Add position embeddings
            position_ids = np.arange(seq_length)[None, :].repeat(batch_size, axis=0)
            for b in range(batch_size):
                for s in range(seq_length):
                    pos_id = position_ids[b, s]
                    if pos_id < self.max_seq_length:
                        hidden_states[b, s] += self.position_embeddings[pos_id]
                        
        if profile:
            profiling_data['embedding_time'] = time.time() - embed_start
            
        # Process attention mask
        causal_mask = np.triu(
            np.ones((seq_length, seq_length), dtype=np.float32) * -1e9, 
            k=1
        )

        # Expand causal mask for batch and heads dimensions
        expanded_causal_mask = causal_mask[None, None, :, :]
            
        if attention_mask is not None:
            # Expand attention mask [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            expanded_attn_mask = attention_mask[:, None, None, :]
            # Convert 0s to -inf, 1s to 0
            expanded_attn_mask = (1.0 - expanded_attn_mask) * -1e9
            # Expand to match the shape expected for broadcasting
            attn_mask = expanded_causal_mask + expanded_attn_mask
        else:
            attn_mask = expanded_causal_mask
            
        # Get sparse attention pattern if enabled
        if self.config["sparse_attention"] and self.sparse_attention:
            sparse_mask = self.sparse_attention.get_sparse_mask(
                seq_length,
                local_window=self.config["local_attention_window"],
                global_stride=self.config["global_attention_stride"]
            )
            # Prepare sparse mask for broadcasting (add batch and head dimensions)
            sparse_mask = sparse_mask[None, None, :, :]
            # Combine with attention mask
            attn_mask = attn_mask + (1.0 - sparse_mask) * -1e9
            
        # Clear KV cache for new sequences if needed
        if use_kv_cache:
            if seq_length > 1:
                self.kv_cache.clear()
                
        # Process each transformer layer
        for i, layer in enumerate(self.layers):
            if profile:
                layer_start = time.time()
                
            # Layer norm 1
            if self.config["use_numba"]:
                ln_1_output = _layer_norm_numba(
                    hidden_states,
                    layer['ln_1_weight'],
                    layer['ln_1_bias'],
                    self.config["layer_norm_epsilon"]
                )
            else:
                # Standard layer norm
                ln_1_output = self._layer_norm(
                    hidden_states,
                    layer['ln_1_weight'],
                    layer['ln_1_bias']
                )
                
            # Self-attention
            if profile:
                attn_start = time.time()
                
            # Compute query, key, value projections
            if self.config["quantize"]:
                # Dequantize weights first
                query_weight = self.quantizer.dequantize_weight(
                    layer['query'], f'layer_{i}_query'
                )
                key_weight = self.quantizer.dequantize_weight(
                    layer['key'], f'layer_{i}_key'
                )
                value_weight = self.quantizer.dequantize_weight(
                    layer['value'], f'layer_{i}_value'
                )
                
                # Matrix multiplications
                if self.config["use_numba"]:
                    query = np.empty(
                        (batch_size, seq_length, self.n_heads * self.head_size),
                        dtype=np.float32
                    )
                    key = np.empty(
                        (batch_size, seq_length, self.n_heads * self.head_size),
                        dtype=np.float32
                    )
                    value = np.empty(
                        (batch_size, seq_length, self.n_heads * self.head_size),
                        dtype=np.float32
                    )
                    
                    for b in range(batch_size):
                        _matmul_numba_parallel(
                            ln_1_output[b], query_weight, query[b]
                        )
                        _matmul_numba_parallel(
                            ln_1_output[b], key_weight, key[b]
                        )
                        _matmul_numba_parallel(
                            ln_1_output[b], value_weight, value[b]
                        )
                else:
                    query = ln_1_output @ query_weight
                    key = ln_1_output @ key_weight
                    value = ln_1_output @ value_weight
            else:
                # Standard matrix multiplications
                if self.config["use_numba"]:
                    query = np.empty(
                        (batch_size, seq_length, self.n_heads * self.head_size),
                        dtype=np.float32
                    )
                    key = np.empty(
                        (batch_size, seq_length, self.n_heads * self.head_size),
                        dtype=np.float32
                    )
                    value = np.empty(
                        (batch_size, seq_length, self.n_heads * self.head_size),
                        dtype=np.float32
                    )
                    
                    for b in range(batch_size):
                        _matmul_numba_parallel(
                            ln_1_output[b], layer['query'], query[b]
                        )
                        _matmul_numba_parallel(
                            ln_1_output[b], layer['key'], key[b]
                        )
                        _matmul_numba_parallel(
                            ln_1_output[b], layer['value'], value[b]
                        )
                else:
                    query = ln_1_output @ layer['query']
                    key = ln_1_output @ layer['key']
                    value = ln_1_output @ layer['value']
                    
            # Add bias if configured
            if self.config["use_bias"]:
                if 'query_bias' in layer:
                    query = query + layer['query_bias']
                if 'key_bias' in layer:
                    key = key + layer['key_bias']
                if 'value_bias' in layer:
                    value = value + layer['value_bias']
                    
            # Reshape for multi-head attention
            query = np.reshape(
                query, 
                (batch_size, seq_length, self.n_heads, self.head_size)
            )
            query = np.transpose(query, (0, 2, 1, 3))  # [batch, heads, seq_len, head_size]
            
            # KV caching for efficient inference
            if use_kv_cache and seq_length == 1 and self.kv_cache.current_seq_length > 0:
                # We're generating tokens one by one after an initial prompt
                # Reshape new key and value
                new_key = np.reshape(
                    key, 
                    (batch_size, 1, self.n_heads, self.head_size)
                )
                new_key = np.transpose(new_key, (0, 2, 1, 3))
                
                new_value = np.reshape(
                    value, 
                    (batch_size, 1, self.n_heads, self.head_size)
                )
                new_value = np.transpose(new_value, (0, 2, 1, 3))
                
                # Update KV cache
                self.kv_cache.update(i, new_key, new_value)
                
                # Get cached keys and values
                key, value = self.kv_cache.get(i, batch_size)
            else:
                # Standard processing
                key = np.reshape(
                    key, 
                    (batch_size, seq_length, self.n_heads, self.head_size)
                )
                key = np.transpose(key, (0, 2, 1, 3))
                
                value = np.reshape(
                    value, 
                    (batch_size, seq_length, self.n_heads, self.head_size)
                )
                value = np.transpose(value, (0, 2, 1, 3))
                
                # Update KV cache for future tokens
                if use_kv_cache:
                    self.kv_cache.update(i, key, value)
                    
            # Compute attention scores
            attention_scores = np.matmul(
                query, 
                np.transpose(key, (0, 1, 3, 2))
            )
            
            # Scale attention scores
            attention_scores = attention_scores / np.sqrt(self.head_size)
            
            # Apply attention mask
            kv_len = key.shape[2]
            # Slice the mask to match the key length (important for KV cache cases)
            current_attn_mask = attn_mask[:batch_size, :, :seq_length, :kv_len]
            attention_scores = attention_scores + current_attn_mask

            #attention_scores = attention_scores + attn_mask[:batch_size, :, :seq_length, :kv_len]
            
            # Apply softmax
            if self.config["use_numba"]:
                attention_probs = np.zeros_like(attention_scores)
                for b in range(batch_size):
                    for h in range(self.n_heads):
                        attention_probs[b, h] = _softmax_numba(attention_scores[b, h])
            else:
                # Standard softmax
                attention_scores_max = np.max(
                    attention_scores, 
                    axis=-1, 
                    keepdims=True
                )
                attention_scores_exp = np.exp(attention_scores - attention_scores_max)
                attention_probs = attention_scores_exp / np.sum(
                    attention_scores_exp, 
                    axis=-1, 
                    keepdims=True
                )
                
            # Apply attention to values
            context = np.matmul(attention_probs, value)
            
            # Reshape back
            context = np.transpose(context, (0, 2, 1, 3))
            context = np.reshape(
                context, 
                (batch_size, seq_length, self.n_heads * self.head_size)
            )
            
            if profile:
                profiling_data[f'layer_{i}_attention'] = time.time() - attn_start
                ffn_start = time.time()
                
            # Project back to hidden size
            if self.config["quantize"]:
                # Dequantize weights
                attn_output_weight = self.quantizer.dequantize_weight(
                    layer['attn_output'], f'layer_{i}_attn_output'
                )
                
                # Matrix multiplication
                if self.config["use_numba"]:
                    attn_output = np.empty(
                        (batch_size, seq_length, self.hidden_size),
                        dtype=np.float32
                    )
                    for b in range(batch_size):
                        _matmul_numba_parallel(
                            context[b], attn_output_weight, attn_output[b]
                        )
                else:
                    attn_output = context @ attn_output_weight
            else:
                # Standard matrix multiplication
                if self.config["use_numba"]:
                    attn_output = np.empty(
                        (batch_size, seq_length, self.hidden_size),
                        dtype=np.float32
                    )
                    for b in range(batch_size):
                        _matmul_numba_parallel(
                            context[b], layer['attn_output'], attn_output[b]
                        )
                else:
                    attn_output = context @ layer['attn_output']
                    
            # Add bias if configured
            if self.config["use_bias"] and 'attn_output_bias' in layer:
                attn_output = attn_output + layer['attn_output_bias']
                
            # First residual connection
            attn_output = hidden_states + attn_output
            
            # Layer norm 2
            if self.config["use_numba"]:
                ln_2_output = _layer_norm_numba(
                    attn_output,
                    layer['ln_2_weight'],
                    layer['ln_2_bias'],
                    self.config["layer_norm_epsilon"]
                )
            else:
                ln_2_output = self._layer_norm(
                    attn_output,
                    layer['ln_2_weight'],
                    layer['ln_2_bias']
                )
                
            # Feed-forward network
            if self.config["quantize"]:
                # Dequantize weights
                fc1_weight = self.quantizer.dequantize_weight(
                    layer['fc_1'], f'layer_{i}_fc_1'
                )
                
                # Matrix multiplication
                if self.config["use_numba"]:
                    fc1_output = np.empty(
                        (batch_size, seq_length, self.intermediate_size),
                        dtype=np.float32
                    )
                    for b in range(batch_size):
                        _matmul_numba_parallel(
                            ln_2_output[b], fc1_weight, fc1_output[b]
                        )
                else:
                    fc1_output = ln_2_output @ fc1_weight
            else:
                # Standard matrix multiplication
                if self.config["use_numba"]:
                    fc1_output = np.empty(
                        (batch_size, seq_length, self.intermediate_size),
                        dtype=np.float32
                    )
                    for b in range(batch_size):
                        _matmul_numba_parallel(
                            ln_2_output[b], layer['fc_1'], fc1_output[b]
                        )
                else:
                    fc1_output = ln_2_output @ layer['fc_1']
                    
            # Add bias if configured
            if self.config["use_bias"] and 'fc_1_bias' in layer:
                fc1_output = fc1_output + layer['fc_1_bias']
                
            # Apply activation function
            if self.config["activation"] == "gelu":
                if self.config["use_numba"]:
                    fc1_output = _gelu_numba(fc1_output)
                else:
                    # Standard GELU
                    fc1_output = 0.5 * fc1_output * (1.0 + np.tanh(
                        np.sqrt(2.0 / np.pi) * (fc1_output + 0.044715 * fc1_output**3)
                    ))
            elif self.config["activation"] == "relu":
                fc1_output = np.maximum(0, fc1_output)
                
            # Second FFN layer
            if self.config["quantize"]:
                # Dequantize weights
                fc2_weight = self.quantizer.dequantize_weight(
                    layer['fc_2'], f'layer_{i}_fc_2'
                )
                
                # Matrix multiplication
                if self.config["use_numba"]:
                    fc2_output = np.empty(
                        (batch_size, seq_length, self.hidden_size),
                        dtype=np.float32
                    )
                    for b in range(batch_size):
                        _matmul_numba_parallel(
                            fc1_output[b], fc2_weight, fc2_output[b]
                        )
                else:
                    fc2_output = fc1_output @ fc2_weight
            else:
                # Standard matrix multiplication
                if self.config["use_numba"]:
                    fc2_output = np.empty(
                        (batch_size, seq_length, self.hidden_size),
                        dtype=np.float32
                    )
                    for b in range(batch_size):
                        _matmul_numba_parallel(
                            fc1_output[b], layer['fc_2'], fc2_output[b]
                        )
                else:
                    fc2_output = fc1_output @ layer['fc_2']
                    
            # Add bias if configured
            if self.config["use_bias"] and 'fc_2_bias' in layer:
                fc2_output = fc2_output + layer['fc_2_bias']
                
            if profile:
                profiling_data[f'layer_{i}_ffn'] = time.time() - ffn_start
                
            # Second residual connection
            hidden_states = attn_output + fc2_output
            
            if profile:
                profiling_data[f'layer_{i}_total'] = time.time() - layer_start
                
        # Final layer norm
        if self.config["use_numba"]:
            hidden_states = _layer_norm_numba(
                hidden_states,
                self.ln_f['weight'],
                self.ln_f['bias'],
                self.config["layer_norm_epsilon"]
            )
        else:
            hidden_states = self._layer_norm(
                hidden_states,
                self.ln_f['weight'],
                self.ln_f['bias']
            )
            
        if profile:
            lm_head_start = time.time()
            
        # Compute logits - only need last token for generation
        if self.config["quantize"]:
            # Get only the last token for generation
            last_hidden = hidden_states[:, -1:, :]
            
            # Dequantize embeddings for logits computation
            scale = self.quantizer.scales['token_embeddings']
            zero_point = self.quantizer.zero_points['token_embeddings']
            dequant_embeddings = (self.token_embeddings.astype(np.float32) - zero_point) * scale
            
            # Matrix multiplication 
            logits = last_hidden @ dequant_embeddings.T
        else:
            # Standard matrix multiplication with embeddings
            last_hidden = hidden_states[:, -1:, :]
            logits = last_hidden @ self.lm_head.T
            
        if profile:
            profiling_data['lm_head_time'] = time.time() - lm_head_start
            profiling_data['total_time'] = time.time() - start_time
            
            # Store profiling data
            self.profiling = profiling_data
            
        return logits

    def _layer_norm(self, x, weight, bias, eps=None):
        """
        Standard layer normalization.
        
        Args:
            x: Input tensor
            weight: Scale parameter
            bias: Bias parameter
            eps: Epsilon for numerical stability
            
        Returns:
            Normalized tensor
        """
        if eps is None:
            eps = self.config["layer_norm_epsilon"]
            
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(variance + eps)
        return x * weight + bias
        
    def generate(self, input_ids=None, prompt=None, max_new_tokens=20, 
                 temperature=1.0, top_k=50, top_p=0.9, 
                 repetition_penalty=1.0, do_sample=True):
        """
        Generate text auto-regressively.
        
        Args:
            input_ids: Token IDs to use as context
            prompt: Text prompt to use as context
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Dictionary with generated token IDs and text
        """
        # Convert prompt to input_ids if provided
        if prompt is not None and input_ids is None:
            if self.tokenizer is None:
                self.tokenizer = OptimizedTokenizer(vocab_size=self.vocab_size)
                
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = np.array([input_ids])  # Add batch dimension
        elif input_ids is None:
            raise ValueError("Either input_ids or prompt must be provided")
            
        # Ensure input_ids has batch dimension
        if len(input_ids.shape) == 1:
            input_ids = input_ids[np.newaxis, :]
            
        # Generate tokens
        generated_ids = input_ids.copy()
        past_token_ids = generated_ids[0].tolist()  # For repetition penalty
        
        # Reset KV cache for new sequence
        if self.config["use_kv_cache"] and self.kv_cache is not None:
            self.kv_cache.clear()
            
        # Initial forward pass with full context
        _ = self.forward(generated_ids)
        
        # Generate tokens auto-regressively
        for _ in range(max_new_tokens):
            # Get only the last token for next prediction
            curr_input_ids = generated_ids[:, -1:]
            
            # Forward pass
            logits = self.forward(curr_input_ids)[0, 0]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in past_token_ids:
                    logits[token_id] = logits[token_id] / repetition_penalty
                    
            # Get probabilities
            probs = self._softmax(logits)
            
            # Apply top-k sampling
            if top_k > 0:
                top_k = min(top_k, logits.shape[-1])
                indices_to_remove = np.argpartition(probs, -top_k)[:-top_k]
                probs[indices_to_remove] = 0
                
                # Renormalize probabilities
                probs = probs / np.sum(probs)
                
            # Apply nucleus (top-p) sampling
            if top_p < 1.0:
                sorted_probs = np.sort(probs)[::-1]
                sorted_indices = np.argsort(probs)[::-1]
                cumulative_probs = np.cumsum(sorted_probs)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0
                
                # Renormalize probabilities
                probs = probs / np.sum(probs)
                
            # Sample from the probability distribution or use greedy decoding
            if do_sample:
                next_token = np.random.choice(probs.shape[0], p=probs)
            else:
                next_token = np.argmax(probs)
                
            # Append new token
            next_token_array = np.array([[next_token]])
            generated_ids = np.concatenate([generated_ids, next_token_array], axis=1)
            past_token_ids.append(next_token)
            
            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                break
                
        # Decode generated tokens
        if self.tokenizer is not None:
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        else:
            generated_text = "[Decoded text not available - no tokenizer]"
            
        return {
            'generated_ids': generated_ids,
            'generated_text': generated_text
        }
        
    def _softmax(self, x):
        """
        Standard softmax function.
        
        Args:
            x: Input tensor
            
        Returns:
            Softmax probabilities
        """
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)
        
    def export_to_onnx(self, model_path, opset_version=12):
        """
        Export model to ONNX format for deployment.
        
        Args:
            model_path: Path to save ONNX model
            opset_version: ONNX opset version
            
        Returns:
            Path to saved model
        """
        if not self.config["use_onnx"]:
            print("ONNX export is disabled in config")
            return None
            
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print("ONNX export requires onnx and onnxruntime packages")
            return None
            
        print(f"Exporting model to ONNX format (opset {opset_version})...")
        
        # Create a simplified model for export
        # In practice, we'd need a proper ONNX export implementation
        # with dynamic axes support, etc.
        
        # For now, just save a simple note
        onnx_path = f"{model_path}.onnx"
        with open(onnx_path, 'w') as f:
            f.write("ONNX export placeholder")
            
        print(f"Model exported to {onnx_path}")
        return onnx_path
        
    def save_model(self, path):
        """
        Save model weights and configuration to disk.
        
        Args:
            path: Path to save model
            
        Returns:
            Dictionary with saved paths
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model configuration
        config_path = f"{path}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model weights
        weights_path = f"{path}_weights.pkl"
        model_data = {
            'token_embeddings': self.token_embeddings,
            'position_embeddings': self.position_embeddings,
            'layers': self.layers,
            'ln_f': self.ln_f
        }
        
        # Save quantization data if used
        if self.config["quantize"]:
            model_data['quantizer'] = {
                'scales': self.quantizer.scales,
                'zero_points': self.quantizer.zero_points
            }
            
        with open(weights_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        # Export to ONNX if configured
        onnx_path = None
        if self.config["use_onnx"]:
            onnx_path = self.export_to_onnx(path)
            
        print(f"Model saved to {path} (config: {config_path}, weights: {weights_path})")
        
        return {
            'config_path': config_path,
            'weights_path': weights_path,
            'onnx_path': onnx_path
        }
        
    def load_model(self, path):
        """
        Load model weights and configuration from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            True if successful
        """
        # Load configuration
        config_path = f"{path}_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Update model parameters from config
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]
        self.n_layers = self.config["n_layers"]
        self.n_heads = self.config["n_heads"]
        self.head_size = self.config["head_size"]
        self.intermediate_size = self.config["intermediate_size"]
        self.max_seq_length = self.config["max_seq_length"]
        
        # Re-initialize optimization components
        if self.config["quantize"]:
            self.quantizer = AdvancedQuantization(
                quantize_type=self.config["quantize_type"],
                per_channel=self.config["per_channel_quantization"],
                optimize_zero_points=self.config["optimize_zero_points"]
            )
        else:
            self.quantizer = None
            
        if self.config["sparse_attention"]:
            self.sparse_attention = SparseAttentionPattern(
                max_seq_length=self.max_seq_length,
                pattern_type=self.config["sparse_attention_pattern"]
            )
        else:
            self.sparse_attention = None
            
        if self.config["use_kv_cache"]:
            self.kv_cache = OptimizedKVCache(
                max_batch_size=1,  # Default for inference
                max_seq_length=self.max_seq_length,
                num_layers=self.n_layers,
                num_heads=self.n_heads,
                head_dim=self.head_size,
                use_memory_mapping=self.config["use_memory_mapping"]
            )
        else:
            self.kv_cache = None
            
        # Load weights
        weights_path = f"{path}_weights.pkl"
        with open(weights_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.token_embeddings = model_data['token_embeddings']
        self.position_embeddings = model_data['position_embeddings']
        self.layers = model_data['layers']
        self.ln_f = model_data['ln_f']
        
        # Load quantization data if available
        if 'quantizer' in model_data and self.quantizer:
            self.quantizer.scales = model_data['quantizer']['scales']
            self.quantizer.zero_points = model_data['quantizer']['zero_points']
            
        # Align arrays for SIMD operations
        if self.config["use_numba"]:
            self.token_embeddings = align_memory(self.token_embeddings)
            self.position_embeddings = align_memory(self.position_embeddings)
            
            for layer in self.layers:
                for key, value in layer.items():
                    layer[key] = align_memory(value)
                    
            self.ln_f['weight'] = align_memory(self.ln_f['weight'])
            self.ln_f['bias'] = align_memory(self.ln_f['bias'])
            
        # Tie weights
        self.lm_head = self.token_embeddings
        
        # Initialize tokenizer
        self.tokenizer = OptimizedTokenizer(vocab_size=self.vocab_size)
        
        # Load ONNX model if available
        if self.config["use_onnx"]:
            onnx_path = f"{path}.onnx"
            if os.path.exists(onnx_path):
                try:
                    # Create ONNX inference session
                    providers = ['CPUExecutionProvider']
                    self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                    print(f"Loaded ONNX model from {onnx_path}")
                except Exception as e:
                    print(f"Error loading ONNX model: {e}")
                    
        # Calculate memory usage
        self.calculate_memory_usage()
        
        print(f"Model loaded from {path} with {self.get_parameter_count()/1e6:.2f}M parameters")
        print(f"Estimated memory usage: {self.memory_used / (1024**2):.2f} MB")
        
        return True

# Example usage
def run_example():
    # Create a smaller model configuration for CPU
    config = {
        "vocab_size": 32000,
        "hidden_size": 768,
        "n_layers": 6,
        "n_heads": 12,
        "head_size": 64,
        "intermediate_size": 2048,
        "max_seq_length": 512,
        
        # Optimization settings
        "quantize": True,
        "quantize_type": "int8",
        "per_channel_quantization": True,
        "optimize_zero_points": True,
        "sparse_attention": True,
        "sparse_attention_pattern": "block_local_global",
        "local_attention_window": 128,
        "global_attention_stride": 32,
        "use_memory_mapping": True,
        "use_numba": True,
        "use_kv_cache": True,
        "profile_execution": True
    }
    
    # Initialize model
    model = OptimizedCPULLM(config)
    model.initialize_model()
    
    # Create sample input
    sample_text = "Once upon a time, in a land far away,"
    
    # Prepare tokenizer
    if model.tokenizer is None:
        model.tokenizer = OptimizedTokenizer(vocab_size=model.vocab_size)
        
    # Tokenize input
    input_data = model.tokenizer.batch_encode(
        [sample_text], 
        max_length=model.max_seq_length
    )
    
    # Generate text
    generation_result = model.generate(
        input_ids=input_data['input_ids'],
        max_new_tokens=20,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    
    # Print result
    print(f"Input: {sample_text}")
    print(f"Generated: {generation_result['generated_text']}")
    
    # Print profiling information
    if model.profiling:
        print("\nProfiling results:")
        for key, value in model.profiling.items():
            print(f"  {key}: {value*1000:.2f} ms")
            
    # Save model
    model.save_model("output/optimized_cpu_llm")
    
    return model

if __name__ == "__main__":
    run_example()
