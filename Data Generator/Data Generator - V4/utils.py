
# coding: utf-8

# In[ ]:


import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import convolve2d
from skimage.io import imsave, imread
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
from six.moves import cPickle as pickle




def deslocamento(a, b, c, x):
    y = a*np.sin(b+(c*x))
    return y


# In[ ]:


def generate_panel(panel_size, a, T, c, d):
    panel = np.zeros(panel_size)

    #Define curva
    horiz_indices = np.arange(panel_size[1])
    curve, derivative = def_curve(horiz_indices=horiz_indices, window_size=panel_size[1], a=a, T=T, c=c, d=d)
    curve_indices = (np.round(curve)).astype(int)
    vertical_steps = sort_vertical_steps(window_size=panel_size)
    
    #Gera o painel 80x80 com as curvas
    for step in vertical_steps:
        vert_indices = curve_indices + step
        # Considera somente os índices que não ultrapassaram as bordas do painel
        valid_indices = np.where(np.logical_and((vert_indices >= 0), (vert_indices < panel_size[0])))[0]
        reflexivity = random.uniform(-1,1)
        panel[vert_indices[valid_indices], horiz_indices[valid_indices]] = reflexivity
    
    return panel, derivative

def def_curve(horiz_indices, window_size, a, T, c, d):

    b = (T*2*np.pi)/window_size
    
    senoide = a*np.sin(b*horiz_indices + c)
    
    cisalhamento = d*horiz_indices
    
    derivative = a*np.cos(b*horiz_indices + c)*b +d
    
    curve = senoide+cisalhamento
    
    return curve, derivative
  


# In[ ]:


def sort_vertical_steps(window_size=(49,33)):
    List = []
    
    List.append(random.randint(0, 10))
    
    while List[-1] < window_size[0]:
        y = random.randint(5, 15) + List[-1]
        List.append(y)
    
    if List[-1] > window_size[0] - 1:
        List.pop()
        
    return List


# In[ ]:
def possible_angles(radianos, derivative):
    position = len(derivative) // 2
    
    slope = -(np.arctan(derivative[position]))
    if slope < 0:
        slope = slope + np.pi
    
    rad_30 = np.pi/6 

    right_slope = ((slope-rad_30)%np.pi) - (np.pi/2)
    left_slope = ((slope+rad_30)%np.pi) - (np.pi/2)

    if left_slope < right_slope:
        new_radians = [elem for elem in radianos if elem > left_slope and elem < right_slope]
    else:
        new_radians = [elem for elem in radianos if elem > left_slope or elem < right_slope]
        
    return new_radians


# Definição de Td:
# https://wiki.seg.org/wiki/Dictionary:Ricker_wavelet
def calculate_min_offset(wavelet_freq, dt):
    td = np.sqrt(6)/(np.pi*wavelet_freq)
    min_offset = (td)/(2*dt)
    min_offset = np.ceil(min_offset).astype(int)
    return min_offset


# In[ ]:


def add_faulting(angle_rad, panel_a, offset, output_shape=(49, 33), panel_b=None):
   
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
    
    if panel_b is not None:
        window[y_origin, x_origin] = panel_b[y_target,x_target]
    else:
        window[y_origin, x_origin] = panel_a[y_target,x_target]

    return window


# In[ ]:


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

def add_noise(diffractions, filter_sigma, signal_to_noise):
    noise_img = np.random.normal(loc=0.0, size=diffractions.shape)

    filtered_img = gaussian_filter(noise_img, sigma=filter_sigma) 
    max_noise = diffractions.max()*signal_to_noise
    filtered_img = (filtered_img/np.abs(filtered_img).max())*max_noise
    filtered_diff = diffractions + filtered_img
    
    return filtered_diff

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

def pickle_data(pickle_file, data_list, name_list):
	"""Salva (pickle) um conjunto de dados para uso futuro.

	Args:
		pickle_file: nome do arquivo onde o dado será salvo
		data_list: lista com os dados a serem salvos
		name_list: lista com o nome de cada dado
	
	Retorno:
		Nenhum, somente salva o dado em disco.

	Obs: data_list e name_list devem ter o mesmo tamanho.
	"""
	if (len(data_list) != len(name_list)):
		print("Erro: lista de dados e lista de nomes devem ter o mesmo tamanho. Nada foi salvo.")
		return
	# Cria diretório para o arquivo, se necessários
	pickle_dir = os.path.dirname(pickle_file)
	if not os.path.exists(pickle_dir): os.mkdir(pickle_dir)
	# Constrói dicionário que associa cada nome a um dado:
	save_dic = {}
	for data, name in zip(data_list, name_list):
		# Sanity check: verifica se não tem nome repetido na lista de nomes
		if save_dic.has_key(name):
			print("Erro - nome duplicado:", name)
			print("Nada foi salvo.")
			return
		save_dic[name] = data
	# Salva o conjunto de dados
	try:
		f = open(pickle_file, 'wb')
		pickle.dump(save_dic, f, pickle.HIGHEST_PROTOCOL)
		f.close()
		print("Dado persistido em", pickle_file)
	except Exception as e:
		print('Erro: não foi possível salvar o dado em', pickle_file, ':', e)
		raise
        
def view_samples(dataset, labels, label_colors='g',
				num_samples=None, num_columns=5, resize=1.0, color='gray', title="",
				save_fig=None):
	"""Visualiza um conjunto de amostras

	Args:
		dataset: conjunto de amostras
		labels: rótulos das amostras
		label_names: lista com o nome de cada rótulo
		label_colors: lista com a cor de cada rótulo
		num_samples: quantas amostras do dataset deve-se sortear para visualização.
			Se None, visualiza o dataset inteiro
		num_columns: em quantas colunas plotar as amostras
		resize: fator de redimensionamento a imagem. Default: 1.0
		color: que padrão de cores usar na visualização. Deve ser um cmap válido do matplotlib.
					Default: 'gray'.
		title: título da imagem
		save_fig: caminho para salvar a figura como imagem. Se None, mostra a figura
			ao invés de salvá-la.
	
	Retorno:
		Nenhum, somente plota as amostras
	"""
	# Se num_samples estiver especificado, sorteia num_samples amostras 
	if not num_samples is None:
		if num_samples > dataset.shape[0]:
			print("Erro: 'num_samples' deve ser menor ou igual ao tamanho do dataset")
			return
		from sklearn.utils import resample
		dataset, labels = resample(dataset, labels, replace=False, n_samples=num_samples)
	dataset_size = dataset.shape[0]
	half_window_x = dataset.shape[1] // 2
	half_window_y = dataset.shape[2] // 2    
	num_columns = min(num_columns, dataset_size)
	num_lines = int(np.ceil(float(dataset_size)/num_columns))
	vmax = dataset.max()
	vmin = -vmax 
	print("Visualizando '%s' - %d amostras" % (title, dataset_size))
	# Plota as amostras 
	fig_width = resize * num_columns
	fig_height = resize * num_lines
	fig, axarr = plt.subplots(num_lines, num_columns, figsize=(fig_width, fig_height))
	if title != "":
		sup_title = fig.suptitle(title)
	else: sup_title = None
	for i in range(num_lines):
		for j in range(num_columns):
			index = i*num_columns + j
			if (num_lines > 1): ax = axarr[i, j]
			else: ax = axarr[j]
			ax.axis('off')
			if (index < dataset_size): 
				im = ax.imshow(dataset[index,:,:], cmap=color, vmin=vmin, vmax=vmax)
				if labels[index, 0] == 1:
					grad = labels[index, 1] * 180 / np.pi
					img_label = 'Quebra de ' + '{:0.2f}'.format(grad) + ' Graus'
					ax.set_title(img_label)
					# Marca o evento/não-evento no ponto central
					#x = np.linspace(0, (dataset.shape[1]-1), dataset.shape[1])
					#m = np.tan(falt_grade*(np.pi)/180)
					#y = (m*(x-half_window_y))+half_window_x 
					#ax.plot(x, y, c='g')
					#ax.set_ylim([0, (dataset.shape[1]-1)])
				else:
					img_label = 'Classe Negativa'
					ax.set_title(img_label)
	fig.tight_layout()
	if not sup_title is None:
		sup_title.set_y(0.95)
		fig.subplots_adjust(top=0.85)	
	if save_fig: 
		plt.savefig(save_fig)
		plt.close(fig)
		print("Figura salva em", save_fig)
	else: plt.show()
	

def load_pickle(pickle_file):
	"""Carrega um conjunto de dados preservado em um pickle
	
	Args:
		pickle_file: nome do arquivo que contém os dados.

	Retorno:
		Dicionário com pares { nome do dado : conteúdo }
	"""
	with open(pickle_file, 'rb') as f:
		data_dic = pickle.load(f)
	print('Dados carregados de', pickle_file)
	for key in data_dic.keys():
		print(" -", key)
	return data_dic

def load_traces(stack_file):
	"""Carrega todos os traços de um dado sísmico empilhado - extensão .SGY ou .SU e
	ordena por CDP. Imprime também algumas informações sobre o empilhamento

	Args:
		stack_file: nome do arquivo que contém o dado sísmico.

	Retorno:
		traces: lista com os traços ordenados por CDP
	"""

	# Carrega o empilhamento
	try:
		traces = [t for t in pm.load(stack_file)]
		print("Carregando o empilhamento", stack_file)
	except Exception as e:
		print("Não foi possível carregar o empilhamento", stack_file, "\n", e)
		return None

	# Ordena por CDP
	traces.sort(key=lambda trace: trace.cdp)
	
	# Imprime algumas informações
	cdp_min = min(traces, key=lambda trace: trace.cdp).cdp
	cdp_max = max(traces, key=lambda trace: trace.cdp).cdp
	ns = traces[0].ns
	dt = traces[0].dt * 1000
	print(len(traces), "traços:")
	print(" - CDP: [%d - %d]" % (cdp_min, cdp_max))
	print(" - Amostras por traço (ns):", ns) 
	print(" - Tempo entre amostras (dt): %.1f ms" % dt)
	
	return traces

def view_seismic_stack(stack_data, apices=None, intersects=None, nonapices=None, 
						cdp_offset=0.0, dt=1.0, 
						resize=1.0, clip_percent=1.0, vmin=None, vmax=None, 
						color='gray', title="", legend_loc=None):
	"""Visualiza o empilhamento

	Args:
		stack_data: matriz (numpy.array) contendo os dados do empilhamento (traços sem o cabeçalho)
		apices: pontos considerados ápices. Default: None
		intersects: pontos considerados ápices com intersecção. Default: None
		nonapices: pontos considerados não-ápices. Default: None
		cdp_offset: número do primeiro CDP. Default: 0.0
		dt: valor da amostragem (em segundos). Default: 1.0
		resize: fator de redimensionamento a imagem. Default: 1.0
		clip_percent: trunca a amplitude do empilhamento em 'clip_percent' do valor máximo, para efeitos de
					visualização. Deve ser um número no intervalo [0.0, 1.0]. Default: 1.0
		color: que padrão de cores usar na visualização. Deve ser um cmap válido do matplotlib.
					Default: 'gray'.
		vmin: valor mínimo da escala de cor. É multiplicado pelo fator clip_percent. Default: None
		vmax: valor máximo da escala de cor. É multiplicado pelo fator clip_percent. Default: None
		title: título da imagem. Default: ""

	Retorno:
		Nenhum, somente plota o empilhamento

	Obs:
		Se cdp_offset = 0 e dt = 1, plota a imagem com os índices da matriz
	"""
	# Verifica se o 'clip_percent' é válido
	if not (clip_percent >= 0.0 and clip_percent <= 1.0):
		print("Erro: clip_percent deve ser um número entre 0.0 e 1.0")
		return

	# Determina tamanho da figura
	fig = plt.figure()
	dpi = float(fig.get_dpi())
	img_width = resize * (stack_data.shape[1] / dpi)
	img_height = resize * (stack_data.shape[0] / dpi)
	fig.set_size_inches(img_width, img_height)
	print("Tamanho da imagem (em polegadas): %.2f x %.2f" % (img_width, img_height))

	# Plota a imagem
	if not vmax is None:
		vmax = vmax * clip_percent
	else:
		vmax = stack_data.max() * clip_percent
	if not vmin is None:
		vmin = vmin * clip_percent
	else:
		vmin = stack_data.min() * clip_percent
	plot_coord = [cdp_offset, stack_data.shape[1] + cdp_offset, stack_data.shape[0] * dt, 0]
	plt.imshow(stack_data, cmap=color, aspect='auto', vmin=vmin, vmax=vmax, extent=plot_coord)
	plt.colorbar()

	# Plota pontos, se houver
	if not apices is None:
		apices_cdp_indices = [cdp for cdp, _ in apices]
		apices_time_indices = [time for _, time in apices]
		plt.scatter(apices_cdp_indices, apices_time_indices, c='g', marker='.', label='apices picks')
	if not intersects is None:
		intersects_cdp_indices = [cdp for cdp, _ in intersects]
		intersects_time_indices = [time for _, time in intersects]
		plt.scatter(intersects_cdp_indices, intersects_time_indices, c='y', marker='.', label='intersects picks')
	if not nonapices is None:
		nonapices_cdp_indices = [cdp for cdp, _ in nonapices]
		nonapices_time_indices = [time for _, time in nonapices]
		plt.scatter(nonapices_cdp_indices, nonapices_time_indices, c='r', marker='.', label='nonapices picks')
	# Ativa a legenda
	if not (apices is None and intersects is None and nonapices is None) and legend_loc:
		plt.legend(loc=legend_loc)

	# Trata eventos de clique
	def onclick(mouse_event):
		button = mouse_event.button
		cdp = mouse_event.xdata
		time = mouse_event.ydata
		if (button == 1 or button == 3):
			event = max(2.0-button, 0.0)
			print('evento=%.1f, cdp=%f, tempo=%f' % (event, cdp, time))
		elif button == 2:
			print("Coord: %d, %d" % (int(cdp), int(time)))	
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	# Insere nomes e mostra a figura
	plt.xlabel('CMP')
	plt.ylabel('Time (s)')
	if title != "": plt.title(title)
	#plt.tight_layout(pad=pad)
	plt.show()
	return

def generate_samples(stack, window_size, picks_coord, label=0, num_classes=17, verbose=True):
	"""Extrai as amostras rotuladas do empilhamento.

	Args:
		stack: empilhamento (numpy.array)
		window_size: tamanho da janela ao redor do pick
		picks_coord: coordenadas da matriz referente ao rótulo
		label: rótulo (eventou ou não evento)
		num_classes: número de classes do problema
		flip: se deve ou não duplicar cada amostra usando espelhamento

	Retorno:
		dataset: conjunto de amostras (janelas) centradas nas picks_coords
		labels: conjunto de rótulos referentes a cada janela, no formato one-hot
	"""
	half_size_r = window_size[0] // 2 # Divisão inteira
	half_size_c = window_size[1] // 2
	dataset_size = len(picks_coord)
	# Determina formato para dado 2D ou 3D
	if (len(stack.shape) == 2): shape = (dataset_size, window_size[0], window_size[1])
	else: shape = (dataset_size, window_size[0], window_size[1], stack.shape[-1])
	dataset = np.zeros(shape, dtype=np.float32)
	labels = np.zeros((dataset_size, num_classes), dtype=np.float32)
	# Transforma o label no formato one-hot
	labels[:,label] = 1.0
	# Extrai janelas centradas nas picks_coords
	i = 0
	for [time_index, trace_index] in picks_coord:
		# Caso pick esteja muito próximo da borda, copia apenas o range apropriado
		data_rmin = max(0, half_size_r - time_index)
		data_rmax = min(window_size[0], half_size_r + stack.shape[0] - time_index)
		data_cmin = max(0, half_size_c - trace_index) 
		data_cmax = min(window_size[1], half_size_c + stack.shape[1] - trace_index)
		tmin = time_index - half_size_r
		stack_rmin = max(0, tmin)
		stack_rmax = min(tmin + window_size[0], stack.shape[0])
		tmin = trace_index - half_size_c
		stack_cmin = max(0, tmin)
		stack_cmax = min(tmin + window_size[1], stack.shape[1])
		dataset[i,data_rmin:data_rmax,data_cmin:data_cmax] = stack[stack_rmin:stack_rmax, stack_cmin:stack_cmax]
		print(stack[stack_rmin:stack_rmax, stack_cmin:stack_cmax])
		#print(i)
		#print("data_rmin: %s, data_rmax: %s, data_cmin: %s, data_cmax: %s" % (data_rmin, data_rmax, data_cmin, data_cmax))
		#print("stack_rmin: %s, stack_rmax: %s, stack_cmin: %s, stack_cmax: %s" % (stack_rmin, stack_rmax, stack_cmin, stack_cmax))
		i += 1

	return dataset, labels
