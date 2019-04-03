# -*- coding: utf-8 -*-
# Define diversas funções de apoio para limpar os notebooks

from __future__ import print_function
import numpy as np
import pymei as pm
import matplotlib.pyplot as plt
import os
from six.moves import cPickle as pickle


def load_picks(picks_file, time_scale='s'):
	"""Carrega os picks (eventos ou não eventos) de um empilhamento e retorna uma 
		lista de picks no formato (CDP, time (s))
	
	Args:
		picks_file: nome do arquivo que contém as coordenadas dos picks. Assume que cada pick
			no arquivo de texto está na ordem (CDP, tempo)
		time_scale: escala de tempo dos picks. Pode ser:
			's': segundo
			'ms': milissegundo
			'mus': microssegundo

	Retorno:
		picks_list: lista de picks no formato (CDP, tempo), onde tempo está sempre em segundos
			e CDP é um número inteiro
	"""
	if time_scale == 's':
		time_scale = 1.0
	elif time_scale == 'ms':
		time_scale = 1.0e3
	elif time_scale == 'mus':
		time_scale = 1.0e6
	else:
		print("Escala de tempo não definida, picks não carregados.")
		return None

	# Carrega os picks do arquivo
	picks = np.loadtxt(picks_file)

	# Acumula picks em uma lista, reescalando o tempo para segundo
	picks_list = []
	for cdp, time in picks:
		time = time / time_scale 
		cdp = int(cdp)
		picks_list.append([cdp, time])
	return picks_list



def read_picks(picks_file, event_type=None):
    """
    Lê um arquivo de picks onde cada pick está no formato:
        evento=<0.0/1.0>, cdp=<num_cdp>, tempo=<tempo>
    e retorna uma lista de tuplas (cdp, tempo) ou (event_type, cdp, tempo).
    
    Args:
        picks_file: string com o nome do arquivo de picks.
        event_type: None, 0 ou 1.
            se event_type == 0: retorna lista de picks (cdp, tempo) cujos eventos eram '0.0'
            se event_type == 1: retorna lista de picks (cdp, tempo) cujos eventos eram '1.0'
            se event_type == None: retorna lista de picks (event_type, cdp, tempo)
            
    Retorno:
        picks: lista de picks cujo cada elemento é uma tupla (cdp, tempo) ou (event_type, cdp, tempo)
    """
    # Confere se evento é válido
    if not (event_type is None or event_type == 0 or event_type == 1):
        print("Erro: argumento 'event_type' deve ser None, 0 ou 1.")
        return []
    # Carrega arquivo e extrai picks
    picks = []
    with open(picks_file, 'r') as f:
        for line in f:
            # Extrai informações da string: evento, tempo e cdp
            event_ind = line.find('evento=') + 7
            event = float(line[event_ind:event_ind+3])
            time_ind = line.find('tempo=') + 6
            time = float(line[time_ind:])
            cdp_ind = line.find('cdp=') + 4
            cdp = np.floor(float(line[cdp_ind:time_ind-8]))
            # Constrói a tupla de acordo com argumento 'event_type'
            if event_type is None:
                pick = (event, cdp, time)
            elif event_type == event:
                pick = (cdp, time)
            else:
                pick = None
            # Insere pick na lista
            if not pick is None:
                picks.append(pick)
            
    return picks



def save_picks(picks_file, picks, time_scale='s'):
	"""Salva lista picks (CDP, tempo (s)) no formato txt
	
	Args:
		- picks_file: nome do arquivo que os picks serão salvos
		- picks: lista de pontos no formato (CDP, tempo (s))
		- time_scale: escala de tempo dos picks. Pode ser:
			's': segundo
			'ms': milissegundo
			'mus': microssegundo

	Retorno:
		- Nenhum, somente salva os picks
	"""
	# Cria diretório para o arquivo, se necessários
	picks_dir = os.path.dirname(picks_file)
	if not os.path.exists(picks_dir): os.mkdir(picks_dir)
	with open(picks_file, 'w') as f:
		for cdp, t in picks:
			line = str(cdp) + '\t' + str(t) + '\n'
			f.write(line)
	print("Picks salvos em", picks_file)



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



def write_traces(stack_file, trace_list, stack_image=None):
	"""Escreve traços em arquivo SU. Se houver uma matriz de dados stack_image, 
	combina o cabeçalho de cada traço de trace_list com o dado da coluna correspondente
	de stack_image.

	Args:
		- stack_file: nome do arquivo no qual o dado será salvo.
		- trace_list: lista com traços.
		- stack_image: matriz numpy com dados.

	Obs: se for instanciada uma stack_image, o número de colunas da matriz deve
	corresponder ao número de traços de trace_list e o tamanho de cada coluna deve
	corresponder ao tamanho do dado de cada traço.

	Retorno:
		- nenhum, somente salva o arquivo.
	"""
	
	su_writer = pm.SU(stack_file, mode='wb')
	# Se houver uma stack_image, copia para os traços antes de escrever
	if not stack_image is None:
		# Testa se há mesmo número de traços e de colunas
		if (len(trace_list) != stack_image.shape[1]):
			print("Erro: colunas de stack_image devem corresponder ao número de traços\
			em trace_list")
			return
		# Testa se há mesmo número de amostras na matriz e nos traços
		if (trace_list[0].ns != stack_image.shape[0]):
			print("Erro: número de amostras em stack_image deve ser igual a de cada traço")
			return
		for i in range(len(trace_list)):
			header = trace_list[i].header
			data = stack_image[:,i].astype(np.float32)
			su_writer.writeTrace(pm.Trace(header, data, 0)) 
	else: 
		for t in trace_list:
			su_writer.writeTrace(t)
	print("Dados escritos em", stack_file)
	


def picks_to_index(picks_list, traces):
	"""Transforma uma lista de picks em uma lista de coordenadas da matriz
	
	Args:
		picks_list: lista com os picks no formato (tempo (s), CDP)
		traces: todos os traços carregados em um empilhamento. Assume que está ordenado por CDP.

	Retorno:
		picks_coord: coordenada dos picks como índices de uma matriz, onde cada coordenada
			está  no formato (timde_index, cdp_index)
	"""
	# Constrói dicionário que associa um cdp a um índice (pressupõe empilhamento ordenado por CDP)
	cdp_dic = {}
	for index, trace in enumerate(traces):
		# Sanity check: verifica se não há cdp repetido no empilhamento
		cdp = trace.cdp
		if (cdp_dic.has_key(cdp)): print("Atenção: CDP %d repetido no empilhamento." % cdp)
		cdp_dic[cdp] = index
	
	# Transforma os picks em coordenadas
	picks_coord = []
	dt = traces[0].dt	   # intervalo entre amostras (pymei converte sempre para segundos)
	for cdp, time in picks_list:
		time_index = int((time / dt))
		cdp_index = cdp_dic[int(cdp)]
		picks_coord.append((time_index, cdp_index))
	return picks_coord



def index_to_picks(index_list, cdp_offset, dt):
	"""Transforma uma lista de índices (coordenadas da matriz no formato 
		[(i1,j1),...,(iN,jN)]) em uma lista de picks no formato (CDP, tempo(s)).
	
	Args:
		index_list: lista de índices no formato [(i1,j1),...,(iN,jN)].
		cdp_offset: número do primeiro CDP.
		dt: taxa de amostragem temporal (em segundos).

	Retorno:
		picks: lista no formato (CDP, tempo(s)).
	"""
	index_list = np.array(index_list, dtype=np.float32)
	picks = np.empty(index_list.shape, dtype=np.float32)
	# Transforma os índices (i,j) em picks (CDP, tempo(s))
	picks[:,0] = index_list[:,1] + cdp_offset
	picks[:,1] = index_list[:,0] * dt
	return picks



def get_data_from_traces(trace_list, dtype=np.float32):
	"""Extrai o dado de cada traço e constrói uma matriz numpy de dimensões
	(número de amostras do traço, número de traços).

	Args:
		- trace_list: lista com os traços carregados.
		- dtype: tipo do dado da matriz. Default: float32

	Retorno:
		- matriz numpy onde cada coluna corresponde ao dado de cada traço.
	"""
	return np.array([t.data for t in trace_list], dtype=dtype).T
	
def normalize_as_pic(stack, verbose=True):
	"""Normaliza o empilhamento para o intervalo [0.0 - 1.0] 

	Args:
		stack: numpy.array contendo os dados do empilhamento (sem cabeçalho)
		verbose: se gostaria de imprimir informações pré e pós empilhamento.
				Default: True

	Retorno:
		stack: dado normalizado
	"""
	if (verbose):
		print("Características pré-normalização:")
		print(" - mínimo:", stack.min())
		print(" - máximo:", stack.max())
		print(" - média:", stack.mean())
		print(" - desvio padrão:", stack.std())
	
	# Mapeia linearmente o dado do intervalo [data.min, data.max] para [0.0, 1.0]
	minimum = np.min(stack)
	maximum = np.max(stack)
	stack = (stack - minimum) / (maximum - minimum)

	if (verbose):
		print("\nCaracterísticas pós-normalização:")
		print(" - mínimo:", stack.min())
		print(" - máximo:", stack.max())
		print(" - média:", stack.mean())
		print(" - desvio padrão:", stack.std())
	return stack



def rms_normalize(stack, verbose=True):
	"""Normaliza o empilhamento traço a traço, a partir do RMS de cada traço.
	Obs: esta é uma operação destrutiva, então se não quiser corromper o dado original, 
	deve-se passar uma cópia do empilhamento.

	Args:
		stack: numpy.array contendo os dados do empilhamento (sem cabeçalho)
		verbose: se gostaria de imprimir informações pré e pós empilhamento.
				Default: True

	Retorno:
		stack: dado normalizado
	"""
	if (verbose):
		print("Características pré-normalização:")
		print(" - mínimo:", stack.min())
		print(" - máximo:", stack.max())
		print(" - média:", stack.mean())
		print(" - desvio padrão:", stack.std())
	# Para cada traço no empilhamento, calcula o RMS do traço e divide o traço por este valor
	for j in xrange(stack.shape[1]):
		rms = np.sqrt(np.mean(np.power(stack[:,j], 2)))
		stack[:,j] = stack[:,j] / rms
	# Centra o empilhamento na média
	# stack = stack - stack.mean()
	if (verbose):
		print("\nCaracterísticas pós-normalização:")
		print(" - mínimo:", stack.min())
		print(" - máximo:", stack.max())
		print(" - média:", stack.mean())
		print(" - desvio padrão:", stack.std())
	return stack



def generate_samples(stack, window_size, picks_coord, label=0, num_classes=2, flip=True, verbose=True):
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
	if (verbose): print("Espelhamento das amostras (flip):", flip)
	half_size = window_size // 2 # Divisão inteira
	dataset_size = len(picks_coord)
	if (flip): dataset_size = dataset_size * 2
	# Determina formato para dado 2D ou 3D
	if (len(stack.shape) == 2): shape = (dataset_size, window_size, window_size)
	else: shape = (dataset_size, window_size, window_size, stack.shape[-1])
	dataset = np.zeros(shape, dtype=np.float32)
	labels = np.zeros((dataset_size, num_classes), dtype=np.float32)
	# Transforma o label no formato one-hot
	labels[:,label] = 1.0
	# Extrai janelas centradas nas picks_coords
	i = 0
	for [time_index, trace_index] in picks_coord:
		# Caso pick esteja muito próximo da borda, copia apenas o range apropriado
		data_rmin = max(0, half_size - time_index)
		data_rmax = min(window_size, half_size + stack.shape[0] - time_index)
		data_cmin = max(0, half_size - trace_index) 
		data_cmax = min(window_size, half_size + stack.shape[1] - trace_index)
		tmin = time_index - half_size
		stack_rmin = max(0, tmin)
		stack_rmax = min(tmin + window_size, stack.shape[0])
		tmin = trace_index - half_size
		stack_cmin = max(0, tmin)
		stack_cmax = min(tmin + window_size, stack.shape[1])
		dataset[i,data_rmin:data_rmax,data_cmin:data_cmax] = \
			stack[stack_rmin:stack_rmax, stack_cmin:stack_cmax]
		if (flip): 
			dataset[i+1] = np.flip(dataset[i], 1)
			i += 1
		i += 1

	return dataset, labels



def prepare_data_for_network(data, window_size, padding='VALID', verbose=True):
	"""Adiciona (possivelmente) padding no dado. Caso dado não tenha volume, remodela
	para ter volume 1.

	Args:
		data: dado a ser remodelado (numpy.array)
		window_size: tamanho da janela onde será feita a inferência
		padding: borda de zeros a ser inserida. Valores possíveis:
			- 'SAME': adiciona zeros suficentes para que a saída da rede tenha
					mesma dimensão da entrada.
			- 'VALID': sem padding (janela vai deslizar somente pelo dado válido)

	Retorno:
		data: dado remodelado.
	"""
	# Se dado é 2D, adiciona camada de volume
	if (data.shape[-1] != 1): data = data.reshape(data.shape + (1,))
	# Adiciona padding, se necessário
	if padding == 'SAME':
		# Quanto é adicionado em cada dimensão (pensando em um cubo):
		#   ((pad_acima, pad_abaixo),
		#	(pad_esquerda, pad_direita),
		#	(pad_frente, pad_trás))
		half_window = window_size//2
		# Se a janela tiver tamanho par, as bordas direita e inferior devem ter um
		# pixel a menos que as da esquerda e superior
		if window_size % 2 == 0:
			pad = ( (half_window, half_window-1),
					(half_window, half_window-1),
					(0, 0) )
		# Se tiver tamanho ímpar, o padding é o mesmo em todas as bordas
		else:
			pad = ( (half_window, half_window),
					(half_window, half_window),
					(0, 0) )
		data = np.pad(data, pad, mode='constant')
		print("Adicionado borda de zeros de largura", half_window)
	elif padding == 'VALID':
		print("Nenhuma borda adicionada")
	else:
		print("Padding inválido: valores devem ser 'SAME' ou 'VALID'")
		return None
	print("Novo formato do dado:", data.shape)
	return data



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



def event_picker(stack_data, x_begin=0, y_begin=0, x_end=None, y_end=None, 
			cdp_offset=0, dt=1.0, time_scale='s', 
			resize=1.0, clip_percent=1.0, color='gray',
			vmin=None, vmax=None, 
			save_file=None):
	"""Permite realizar picks de eventos numa janela da imagem
			- Botão esquerdo: dá (cdp, tempo) de evento;
			- Botão direito: dá (cdp, tempo) de não-evento;
			- Botão do meio: dá coordenada do ponto

	Args:
		stack_data: matriz (numpy.array) contendo os dados do empilhamento (traços sem o cabeçalho)
		x,y_begin, x,y_end: início e fim da janela
		cdp_offset: número do primeiro CDP
		dt: valor da amostragem (em segundos)
		time_scale: escala temporal dos picks. Pode ser 's' (segundo) ou 'ms' (milissegundo)
					Default: 's'
		resize: fator de redimensionamento a imagem. Default: 1.0
		clip_percent: trunca a amplitude do empilhamento em 'clip_percent' do valor máximo, para efeitos de
					visualização. Deve ser um número no intervalo [0.0, 1.0]. Default: 1.0
		color: que padrão de cores usar na visualização. Deve ser um cmap válido do matplotlib.
					Default: 'gray'.
		vmin, vmax: valores mínimo e máximo para o mapa de cores
		save_file: se diferente de None, dá append dos picks no arquivo indicado.

	Retorno:
		Nenhum, somente plota o empilhamento

	Obs:
		Se cdp_offset = 0 e dt = 1, plota a imagem com os índices da matriz
	"""
	# Verifica se o 'clip_percent' é válido
	if not (clip_percent >= 0.0 and clip_percent <= 1.0):
		print("Erro: clip_percent deve ser um número entre 0.0 e 1.0")
		return

	# Determina o valor de escala temporal
	if (time_scale=='s'): time_scale = 1.0
	elif (time_scale=='ms'): time_scale = 1.0e3
	else:
		print(time_scale, "não é uma escala temporal válida. Valores possíveis são 's' ou 'ms'")
		return

	# Verifica se janela é válida
	if x_end is None: x_end = stack_data.shape[1]
	if y_end is None: y_end = stack_data.shape[0]
	if (x_begin < 0 or y_begin < 0 or\
		x_end > stack_data.shape[1] or y_end > stack_data.shape[0]):
		print("Erro: coordenada da janela fora das dimensões do empilhamento.")
		return

	# Determina tamanho da figura
	fig = plt.figure()
	dpi = float(fig.get_dpi())
	img_width = resize * ((x_end - x_begin) / dpi)
	img_height = resize * ((y_end - y_begin) / dpi)
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
		vmin = -vmax
	dt_offset = y_begin * dt
	plot_coord = [x_begin+cdp_offset, x_end + cdp_offset, y_end * dt, dt_offset]
	img = stack_data[y_begin:y_end, x_begin:x_end]
	print("Coordenadas:  x - [%.1f, %.1f], y - [%.1f, %.1f]" % 
		(plot_coord[0], plot_coord[1], plot_coord[3], plot_coord[2]))
	print("Dimensão:", img.shape)
	print("")
	plt.imshow(img, cmap=color, aspect='auto', vmin=vmin, vmax=vmax, extent=plot_coord)
	plt.colorbar()

	# Trata eventos de clique
	def onclick(mouse_event):
		button = mouse_event.button
		cdp = mouse_event.xdata
		time = mouse_event.ydata
		time *= time_scale
		if (button == 1 or button == 3):
			event = max(2.0-button, 0.0)
			event_coord = 'evento=%.1f, cdp=%f, tempo=%f' % (event, cdp, time)
			# Faz append das coordenadas no arquvio save_file, se houver
			if not save_file is None:
				with open(save_file, 'a') as f: f.writelines([event_coord,'\n'])
			print(event_coord)
		elif button == 2:
			print("Coord: %d, %d" % (int(cdp), int(time)))	
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	# Insere nomes e mostra a figura
	plt.xlabel('CDP')
	plt.ylabel('Tempo (s)')
	plt.show()
	return


def view_samples(dataset, labels, num_samples=None, num_columns=5, resize=1.0, color='gray', title=""):
	"""Visualiza um conjunto de amostras

	Args:
		dataset: conjunto de amostras
		labels: rótulos das amostras
		num_samples: quantas amostras do dataset deve-se sortear para visualização.
			Se None, visualiza o dataset inteiro
		num_columns: em quantas colunas plotar as amostras
		resize: fator de redimensionamento a imagem. Default: 1.0
		color: que padrão de cores usar na visualização. Deve ser um cmap válido do matplotlib.
					Default: 'gray'.
		title: título da imagem
	
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
	half_window = dataset.shape[1] // 2
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
	label_color='green'
	for i in range(num_lines):
		for j in range(num_columns):
			index = i*num_columns + j
			if (num_lines > 1): ax = axarr[i, j]
			else: ax = axarr[j]
			ax.axis('off')
			if (index < dataset_size): 
				im = ax.imshow(dataset[index,:,:], cmap=color, vmin=vmin, vmax=vmax)
				img_label = labels[index]
				ax.set_title(img_label)
				# Marca o evento/não-evento no ponto central
				ax.scatter([half_window], [half_window], c=label_color, marker='x')
	fig.tight_layout()
	if not sup_title is None:
		sup_title.set_y(0.95)
		fig.subplots_adjust(top=0.85)	
	plt.show()
	


def multiple_event_viewer(stack_data1, stack_data2=None, stack_data3=None, 
				 picks=None, picks_color='g', picks_marker='.',
				 x_begin=0, y_begin=0, x_end=None, y_end=None, 
				 cdp_offset=0, dt=1.0, time_scale='s', 
				 resize=1.0, 
				 clip_percent1=1.0, clip_percent2=1.0, clip_percent3=1.0,
				 color1='seismic', color2='rainbow', color3='seismic',
				 vmin1=None, vmax1=None, 
				 vmin2=None, vmax2=None, 
				 vmin3=None, vmax3=None, 
				 title1='', title2='', title3='',
				 bar1=True, bar2=True, bar3=True,
				 save_file=None):
	"""Visualiza até 03 dados simultaneamente. Permite anotação (picking):
			- Botão esquerdo: dá (cdp, tempo) de evento;
			- Botão direito: dá (cdp, tempo) de não-evento;
			- Botão do meio: dá coordenada do ponto

	Args:
		stack_data1, 2, 3: matriz (numpy.array) contendo os dados do empilhamento (traços sem o cabeçalho)
		picks: lista com pontos anotados no formato (CDP, tempo(s)).
		picks_color: cor para plotar os picks. Default: 'g' (verde).
		picks_marker: marcador para plotar os picks. Default: '.' (ponto).
		x,y_begin, x,y_end: início e fim da janela. Por default mostra a janela inteira.
		cdp_offset: número do primeiro CDP
		dt: valor da amostragem (em segundos)
		time_scale: escala temporal dos picks. Pode ser 's' (segundo) ou 'ms' (milissegundo). Default: 's'
		resize: fator de redimensionamento a imagem. Default: 1.0
		clip_percent1, 2, 3: trunca a amplitude do empilhamento 1, 2 ou 3 em 'clip_percent' 
					do valor máximo, para efeitos de visualização. Deve ser um número 
					no intervalo [0.0, 1.0]. Default: 1.0
		color1, 2, 3: que padrão de cores usar na visualização dos empilhamentos 1, 2 ou 3. 
					Deve ser um cmap válido do matplotlib.
		vmin1, 2, 3, vmax1, 2, 3: valores mínimo e máximo para o mapa de cores de cada empilhamento.
		title1, 2, 3: título do plot de cada empilhamento.
		bar, 2, 3: se True, plota barra de cores (default). Se False, omite a barra de cores.
		save_file: se diferente de None, salva (append) dos picks no arquivo indicado.

	Retorno:
		Nenhum, somente plota o empilhamento

	Obs:
		Se cdp_offset = 0 e dt = 1, plota a imagem com os índices da matriz
	"""
	# Verifica se o 'clip_percent' é válido
	if not (clip_percent1 >= 0.0 and clip_percent1 <= 1.0):
		print("Erro: clip_percent1 deve ser um número entre 0.0 e 1.0")
		return
	if not (clip_percent2 >= 0.0 and clip_percent2 <= 1.0):
		print("Erro: clip_percent2 deve ser um número entre 0.0 e 1.0")
		return
	if not (clip_percent3 >= 0.0 and clip_percent3 <= 1.0):
		print("Erro: clip_percent3 deve ser um número entre 0.0 e 1.0")
		return

	# Determina o valor de escala temporal
	if (time_scale=='s'): ts = 1.0
	elif (time_scale=='ms'): ts = 1.0e3
	else:
		print(time_scale, "não é uma escala temporal válida. Valores possíveis são 's' ou 'ms'")
		return

	# Verifica se janela é válida
	if x_end is None: x_end = stack_data1.shape[1]
	if y_end is None: y_end = stack_data1.shape[0]
	if (x_begin < 0 or y_begin < 0 or\
		x_end > stack_data1.shape[1] or y_end > stack_data1.shape[0]):
		print("Erro: coordenada da janela fora das dimensões do empilhamento.")
		return
	
	# Se picks for definido, extrai cdp e tempo
	if not picks is None:
		# Soma meio passo em cada eixo para que o marcador do pick fique
		# centralizado no pixel
		cdp_indices = [(cdp + 0.5) for cdp, _ in picks]
		time_indices = [(np.floor(t/dt)*dt + dt/2.0) for _, t in picks]

	# Cria figura com a quantidade adequada de subplots 
	num_windows = 1
	if not stack_data2 is None: num_windows += 1
	if not stack_data3 is None: num_windows += 1
	fig, axes = plt.subplots(ncols=num_windows, sharex=True, sharey=True)
	# Computa tamanho da figura
	dpi = float(fig.get_dpi())
	#img_width = (resize*num_windows)* ((x_end - x_begin) / dpi)
	img_width = resize * ((x_end - x_begin) / dpi)
	img_height = resize * ((y_end - y_begin) / dpi)
	fig.set_size_inches(img_width, img_height)
	print("Tamanho da imagem (em polegadas): %.2f x %.2f" % (img_width, img_height))
	
	# Homogeneiza a lista de eixos
	if stack_data2 is None and stack_data3 is None: axes = [axes]
	elif stack_data2 is None: axes = [axes[0], None, axes[1]]

	# Plota a imagem 1
	if not vmax1 is None:
		vmax1 = vmax1 * clip_percent1
	else:
		vmax1 = stack_data1.max() * clip_percent1
	if not vmin1 is None:
		vmin1 = vmin1 * clip_percent1
	else:
		vmin1 = -vmax1
	dt_offset = y_begin * dt
	plot_coord = [x_begin+cdp_offset, x_end + cdp_offset, y_end * dt, dt_offset]
	img = stack_data1[y_begin:y_end, x_begin:x_end]
	print("Coordenadas:  x - [%.1f, %.1f], y - [%.1f, %.1f]" % 
		(plot_coord[0], plot_coord[1], plot_coord[3], plot_coord[2]))
	print("Dimensão:", img.shape)
	print("")
	img1 = axes[0].imshow(img, cmap=color1, aspect='auto', vmin=vmin1, vmax=vmax1, extent=plot_coord, interpolation=None)
	# Se picks for definido, plota picks
	if not picks is None:
		axes[0].scatter(cdp_indices, time_indices, c=picks_color, marker=picks_marker)
	axes[0].set_title(title1)
	if bar1: fig.colorbar(img1, ax=axes[0])

	# Plota a imagem 2
	if not stack_data2 is None:
		if not vmax2 is None:
			vmax2 = vmax2 * clip_percent2
		else:
			vmax2 = stack_data2.max() * clip_percent2
		if not vmin2 is None:
			vmin2 = vmin2 * clip_percent2
		else:
			vmin2 = -vmax2
		img = stack_data2[y_begin:y_end, x_begin:x_end]
		img2 = axes[1].imshow(img, cmap=color2, aspect='auto', vmin=vmin2, vmax=vmax2, extent=plot_coord, interpolation=None)
		# Se picks for definido, plota picks
		if not picks is None:
			axes[1].scatter(cdp_indices, time_indices, c=picks_color, marker=picks_marker)
		axes[1].set_title(title2)
		if bar2: fig.colorbar(img2, ax=axes[1])
		
	# Plota a imagem 3
	if not stack_data3 is None:
		if not vmax3 is None:
			vmax3 = vmax3 * clip_percent3
		else:
			vmax3 = stack_data3.max() * clip_percent3
		if not vmin3 is None:
			vmin3 = vmin3 * clip_percent3
		else:
			vmin3 = -vmax3
		img = stack_data3[y_begin:y_end, x_begin:x_end]
		img3 = axes[2].imshow(img, cmap=color3, aspect='auto', vmin=vmin3, vmax=vmax3, extent=plot_coord, interpolation=None)
		if not picks is None:
			axes[2].scatter(cdp_indices, time_indices, c=picks_color, marker=picks_marker)
		axes[2].set_title(title3)
		if bar3: fig.colorbar(img3, ax=axes[2])
	
	# Trata eventos de clique
	def onclick(mouse_event):
		button = mouse_event.button
		cdp = mouse_event.xdata
		time = mouse_event.ydata
		time *= ts
		event = max(2.0-button, 0.0)
		event_coord = 'evento=%.1f, cdp=%f, tempo=%f' % (event, cdp, time)
		print(event_coord)
		# Faz append das coordenadas no arquvio save_file, se houver
		if not save_file is None:
			with open(save_file, 'a') as f: f.writelines([event_coord,'\n'])
		
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	# Insere nomes e mostra a figura
	fig.text(0.5, 0.000, 'CMP', ha='center')
	fig.text(0.001, 0.5, 'Time (' + time_scale + ')', va='center', rotation='vertical')
	plt.tight_layout()
	plt.show()
	return



def inference_to_picks(inference, cdp_offset, dt, threshold=0.5):
	"""Gera coordenadas no formato (CDP, tempo (s)) a partir de um mapa de inferências.

	Args:
		inference: inferencia a ser analisada
		cdp_offset: número do CDP inicial
		dt: intervalo de tempo 
		threshold: limiar acima do qual será considerado que a inferência foi positiva
					(mapeia 'threshold' para 1.0). Default: 0.5

	Retorno:
		Coordenadas do mapa de inferência nas quais o valor foi acima de 'threshold',
		no formato (CDP, tempo (s)).
	"""
	# Cria cópia para não sobrescrever o dado original
	inference = inference.copy()
	# Limiariza a inferencia
	inference[inference >= threshold] = 1.0
	inference[inference < threshold] = 0.0
	# Extrai as coordenadas de cada inferencia
	ind_row, ind_col = np.where(inference == 1.0)
	return index_to_picks(zip(ind_row, ind_col), cdp_offset, dt)




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
		if name in save_dic:
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



def load_pickle(pickle_file):
	"""Carrega um conjunto de dados preservado em um pickle
	
	Args:
		pickle_file: nome do arquivo que contém os dados.

	Retorno:
		Dicionário com pares { nome do dado : conteúdo }
	"""
	with open(pickle_file, 'rb') as f:
		data_dic = pickle.load(f, encoding='latin1')
	print('Dados carregados de', pickle_file)
	for key in data_dic.keys():
		print(" -", key)
	return data_dic



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



def get_peak_frequency(signal, sample_period, plot_spectrum=False):
	"""Computa a frequência de pico de um sinal. Obs: se o pico estiver em alguma
		das extremidades do espectro (0 ou max_freq), a estimativa será a própria
		extremidade.
	
	Args:
		signal: vetor unidimensional com as amostras do sinal.
		sample_period: tempo de uma amostra (inverso da taxa amostral)
		plot_spectrum: se True, plota o spectro de frequências. Default: False
		
	Retorno:
		peak_frequency: estimativa da frequência de pico do sinal.
	"""
	
	# Computa a FFT do sinal e a magnitude das frequências positivas
	fourier = np.fft.fft(signal)
	frequencies = np.fft.fftfreq(signal.size, sample_period)
	positive_frequencies = frequencies[np.where(frequencies >= 0)]
	magnitudes = abs(fourier[np.where(frequencies >= 0)])
	peak_index = np.argmax(magnitudes)
	# Trata condições de contorno
	if (peak_index == 0) or (peak_index == (magnitudes.size-1)):
		delta = 0.0
	# Computa frequência de pico baseada na estimativa de frequência de MacLeod
	else:
		peaks = np.empty(3, dtype=np.complex128)
		peaks = fourier[peak_index-1:peak_index+2]
		peaks = np.real(peaks*np.conjugate(peaks[1]))
		gamma = (peaks[0]-peaks[2])/(2*peaks[1]+peaks[0]+peaks[2])
		delta = (np.sqrt(1 + 8*gamma*gamma)-1)/(4*gamma)
	peak_frequency = frequencies[peak_index] + delta
	
	if plot_spectrum:
		plt.plot(positive_frequencies, magnitudes)
		plt.grid()
		plt.show()
	
	return peak_frequency



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
		left_taper = np.flip(taper, 0)
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
		left_taper = np.flip(taper, 1)
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
		left_taper = np.flip(taper, 1)
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




