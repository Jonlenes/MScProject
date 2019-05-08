import numpy as np

from util import get_valid_indexs

from scipy.signal import convolve2d


def range_normalize_std(stack, verbose=False, max_std = None):
    """Normaliza o empilhamento para um intervalo de [-1,1] com a opcao de dar clip
    no dado de entrada em relacao ao desvio padrao

    Args:
        stack: numpy.array contendo os dados do empilhamento (sem cabeçalho)
        verbose: se gostaria de imprimir informações pré e pós empilhamento.
            Default: True
        max_std: realiza clip do dado multiplicando esse valor pelo desvio padrao
            Defaut: None

    Retorno:
        stack: dado normalizado
    """
    if (verbose):
        print("Características pré-normalização:")
        print(" - mínimo:", stack.min())
        print(" - máximo:", stack.max())
        print(" - média:", stack.mean())
        print(" - desvio padrão:", stack.std())
    
    #Clip no dado a partir do desvio padrao 
    #stack = clip_std(stack, max_std)
    
    # Normaliza o dado para ficar entre o iteralo de [-1,1]
    mean = np.mean(stack)
    maximum = np.max(stack)
    minimum = np.min(stack)
    stack = stack / max(maximum, abs(minimum))

    if (verbose):
        print("\nCaracterísticas pós-normalização:")
        print(" - mínimo:", stack.min())
        print(" - máximo:", stack.max())
        print(" - média:", stack.mean())
        print(" - desvio padrão:", stack.std())
    return stack


def calculate_min_offset(wavelet_freq, dt):
    td = np.sqrt(6)/(np.pi*wavelet_freq)
    min_offset = (td)/(2*dt)
    min_offset = np.ceil(min_offset).astype(int)
    return min_offset


def add_faulting(angle_rad, panel_a, offset, output_shape=(49, 33)):
    size = panel_a.shape[0]
    center = size//2

    alpha = np.tan(angle_rad+np.pi/2)
    height = output_shape[0]
    width = output_shape[1]
    
    window = np.empty((height,width))

    x_offset = width // 2
    y_offset = height //2
    
    x_begin = center-x_offset
    x_end = x_begin+width
    y_begin = center-y_offset
    y_end = y_begin+height
    
    window = panel_a[y_begin:y_end, x_begin:x_end].copy()


    if angle_rad > -np.pi/4 and angle_rad < np.pi/4:

        y = np.arange(0, height)
        x = (y-y_offset)/alpha + x_offset
        y = np.flip(y, axis=0)
        x_indices = np.round(x).astype(np.int)
        y_indices = np.round(y).astype(np.int)
        valid_indices = np.where(np.logical_and(x_indices>=0, x_indices<width))
        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]

        y_origin = [y for limit, index in zip(x_indices, y_indices) for y in np.tile(index, limit)]
        x_origin = [x for limit in x_indices for x in np.arange(limit)]

        y_slide = offset
        x_slide = np.floor(angle_rad*y_slide).astype(int)

        x_target = np.array(x_origin)-x_slide+x_begin
        y_target = np.array(y_origin)-y_slide+y_begin


    else:
        x = np.arange(0, width)
        y = (height-1) - (alpha*(x-x_offset) + y_offset)
        x_indices = np.round(x).astype(np.int)
        y_indices = np.round(y).astype(np.int)
        valid_indices = np.where(np.logical_and(y_indices>=0, y_indices<height))
        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]

        x_origin = [x for limit, index in zip(y_indices, x_indices) for x in np.tile(index, limit)]
        y_origin = [y for limit in y_indices for y in np.arange(limit)]

        x_slide = offset
        y_slide = np.floor(alpha*x_slide).astype(int)
        x_target = np.array(x_origin)+x_slide+x_begin
        y_target = np.array(y_origin)-y_slide+y_begin
    
    # Jonlenes: added call to get_valid_indexs to avoid 'out of rang' error 
    valid_indexs = get_valid_indexs(y_target, x_target, window.shape[0])
    y_origin = np.array(y_origin)
    x_origin = np.array(x_origin)
    
    window[ y_origin[valid_indexs], x_origin[valid_indexs] ] = panel_a[ y_target[valid_indexs], x_target[valid_indexs] ]

    return window


def ricker_wavelet(frequency, size, dt):
	"""
	Computa a Ricker Wavelet para os tempos e frequência da entrada.
	
	Args:
		frequency: frequência da wavelet.
		size: tamanho do vetor
		dt: taxa de amostragem temporal (em segundos).
		
	Retorno:
		wavelet: vetor de amplitudes da wavelet para cada ponto
	"""
	# Caso tamanho seja par, não existe amostra central
	offset = (size-1)/2 + ((size-1)%2)/2.0
	offset *= dt
	times = np.arange(size, dtype=np.float32) * dt - offset
	cte = -(np.pi**2)*(frequency**2)
	exp = np.exp(cte*(times**2))
	wavelet = exp + 2*cte*(times**2)*exp
	return wavelet

def apply_taper(data, taper_size):
	"""Afina as bordas do dado (tapering) usando a função cosseno. As dimensões que serão afinadas devem ter tamanho
	pelo menos taper_size
	
	Args:
		data: numpy.array de dimensões 1D, 2D ou 3D. Se 1D ou 2D, afunila as bordas. Se 3D, supõe que é uma sequência
			de janelas 2D e afina as bordas de cada janela.
		taper_size: tamanho da borda a ser afinada. Deve ser um inteiro positivo.
		
	Retorno:
		data: dado com as bordas afinadas.
	"""
	# Cria taper com tamanho taper_size no intervalo (1.0, 0.0]
	t = (np.pi / (taper_size)) * (np.arange(taper_size).astype(np.float32) + 1.0)
	taper = (np.cos(t) / 2.0) + 0.5

	# Caso 1D
	if data.ndim == 1:
		data_size = data.shape[0]
		# Sanity check
		if data_size < taper_size:
			print("Erro: dado menor que taper_size. Nenhum taper aplicado.")
			return data
		# Aplica taper na borda esquerda
		left_taper = np.flip(taper, axis=None)
		data[0:taper_size] *= left_taper
		# Aplica taper na borda direita
		right_taper = taper
		data[(data_size-taper_size):] *= right_taper
		
	# Caso 2D
	elif data.ndim == 2:
		data_height = data.shape[0]
		data_width = data.shape[1]
		# Sanity check
		if (data_height < taper_size) or (data_width < taper_size):
			print("Erro: dado menor que taper_size. Nenhum taper aplicado.")
			return data
		# Cria taper de cada borda 
		taper = taper.reshape((1,taper_size))
		right_taper = taper
		left_taper = np.flip(taper, axis=1)
		top_taper = left_taper.T
		bottom_taper = right_taper.T
		# Taper superior
		data[0:taper_size, :] *= top_taper
		# Taper inferior
		data[(data_height - taper_size):, :] *= bottom_taper
		# Taper esquerdo
		data[:, 0:taper_size] *= left_taper
		# Taper direito
		data[:, (data_width - taper_size):] *= right_taper

	# Caso 3D
	elif data.ndim == 3:
		data_depth = data.shape[0]
		data_height = data.shape[1]
		data_width = data.shape[2]
		# Sanity check
		if (data_height < taper_size) or (data_width < taper_size):
			print("Erro: dado menor que taper_size. Nenhum taper aplicado.")
			return data
		# Cria taper de cada borda 
		taper = taper.reshape((1,taper_size))
		right_taper = taper
		left_taper = np.flip(taper)
		top_taper = left_taper.T
		bottom_taper = right_taper.T
		# Aplica taper em cada janela do dado
		for i in range(data_depth):
			# Taper superior
			data[i,0:taper_size, :] *= top_taper
			# Taper inferior
			data[i,(data_height - taper_size):, :] *= bottom_taper
			# Taper esquerdo
			data[i,:, 0:taper_size] *= left_taper
			# Taper direito
			data[i,:, (data_width - taper_size):] *= right_taper
			
	# Caso >3D
	else:
		print("Erro: dado deve ter 1, 2 ou 3 dimensões. Nenhum taper aplicado.")
		return data
	
	return data


def convolve_wavelet(panel, frequency, dt, taper_size, window_size):
    wavelet = ricker_wavelet(frequency, window_size, dt)
    # Constrói wavelet 2D
    wavelet = wavelet.reshape((1,-1)).T
    wavelet_2D = np.empty((window_size, taper_size*2+1), dtype=np.float32)
    wavelet_2D[:,:] = wavelet
    wavelet_2D = apply_taper(wavelet_2D, taper_size)
        
    new_panel = convolve2d(panel, wavelet_2D, mode='same')
    
    return new_panel