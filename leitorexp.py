import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    theta_value = None

    for i in range(26, len(lines)):
        line = lines[i].strip()

        # Parar a leitura ao encontrar "fitted parameters ("
        if "fitted parameters (" in line:
            break

        if line:
            parts = line.split()

            if len(parts) == 6:
                theta_value = float(parts[3])

            elif len(parts) == 4 and theta_value is not None:
                phi = float(parts[0])
                #intensi1 =float(parts[1])
                #intensi2 =float(parts[2])
                #intensitycal = float((intensi1-intensi2)/intensi2)
                intensityexp = float(parts[3])
                data.append([phi, theta_value, intensityexp, True])  # Marcar como original

    df = pd.DataFrame(data, columns=['Phi', 'Theta', 'intensityexp', 'IsOriginal'])
    #resultados, angulos, r_factor_medio, r_factor_total = calcular_r_factor(df)
    # Verificar o intervalo de Phi
    phi_min = df['Phi'].min()
    phi_max = df['Phi'].max()
    phi_interval = phi_max - phi_min

    if phi_interval < 360 and df['Phi'].max() < 360:
        df_360 = df[df['Phi'] == 0].copy()
        df_360['Phi'] = 360
        df = pd.concat([df, df_360], ignore_index=True)

    if phi_interval == 120:
        # Replicação dos dados para cobrir 360 graus
        first_values = df.groupby('Theta').first().reset_index()
        df = df.groupby('Theta', group_keys=False, as_index=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df.groupby('Theta').last().reset_index()  # Pegar os últimos valores
        last_values['Phi'] = first_values['Phi']  # Substituir pelo valor do primeiro Phi original
        # Adicionar os novos valores ao DataFrame
        df = pd.concat([df, last_values], ignore_index=True)

        df_0_120 = df.copy()
        df_0_120['Phi'] = 120 + df_0_120['Phi']
        df_0_120['isOriginal'] = False

        df_240_360 = df.copy()
        df_240_360['Phi'] = 240 + df_240_360['Phi']

        df = pd.concat([df, df_0_120, df_240_360]).reset_index(drop=True)

    if phi_interval == 90:
        # Replicação dos dados para cobrir 360 graus
        first_values = df.groupby('Theta').first().reset_index()
        df = df.groupby('Theta', group_keys=False, as_index=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df.groupby('Theta').last().reset_index()  # Pegar os últimos valores
        last_values['Phi'] = first_values['Phi']  # Substituir pelo valor do primeiro Phi original

        # Adicionar os novos valores ao DataFrame
        df = pd.concat([df, last_values], ignore_index=True)
        df_0_90 = df.copy()
        df_0_90['Phi'] = 90 + df_0_90['Phi']

        df_90_180 = df.copy()
        df_90_180['Phi'] = 180 + df_90_180['Phi']

        df_180_270 = pd.concat([df,df_0_90]).reset_index(drop=True)
        df_180_270['Phi'] = 180 + df_180_270['Phi']

        df = pd.concat([df, df_0_90, df_180_270]).reset_index(drop=True)



    if np.isclose(phi_interval, 48):  # Usei np.isclose por segurança
        print(f"Detectado intervalo de {phi_interval:.1f} graus. Aplicando replicação C8 (8-fold).")
        # Replicação dos dados para cobrir 360 graus
        first_values = df.groupby('Theta').first().reset_index()

            # --- CORREÇÃO 1: Esta linha estava criando um DF vazio ---
            # A sua linha:
            # df = df.groupby('Theta', group_keys=False, as_index=False).apply(lambda x: x.drop(x.index[0]))
            # A forma correta (no seu estilo) de remover o primeiro item de cada grupo:
        df = df.groupby('Theta', group_keys=False).apply(lambda x: x.iloc[1:]).reset_index(drop=True)
            # --- FIM DA CORREÇÃO 1 ---

        last_values = df.groupby('Theta').last().reset_index()  # Pegar os últimos valores
        last_values['Phi'] = first_values['Phi']  # Substituir pelo valor do primeiro Phi original

            # Adicionar os novos valores ao DataFrame
        df = pd.concat([df, last_values], ignore_index=True)

            # (O resto do seu código de replicação C8 está perfeito)

            # Réplica 1 (offset 45)
        df_rep1 = df.copy()
        df_rep1['Phi'] = 45 + df_rep1['Phi']

            # Réplica 2 (offset 90)
        df_rep2 = df.copy()
        df_rep2['Phi'] = 90 + df_rep2['Phi']

            # Réplica 3 (offset 135)
        df_rep3 = df.copy()
        df_rep3['Phi'] = 135 + df_rep3['Phi']

            # Réplica 4 (offset 180)
        df_rep4 = df.copy()
        df_rep4['Phi'] = 180 + df_rep4['Phi']

            # Réplica 5 (offset 225)
        df_rep5 = df.copy()
        df_rep5['Phi'] = 235 + df_rep5['Phi']

            # Réplica 6 (offset 270)
        df_rep6 = df.copy()
        df_rep6['Phi'] = 270 +df_rep6['Phi']

            # Réplica 7 (offset 315)
        df_rep7 = df.copy()
        df_rep7['Phi'] = 315 + df_rep7['Phi']

            # Agora, junte o 'df' original (bloco 1) com todas as 7 réplicas
        df = pd.concat([
            df,  # Bloco 1 (0-44)
            df_rep1,  # Bloco 2 (45-89)
            df_rep2,  # Bloco 3 (90-134)
            df_rep3,  # Bloco 4 (135-179)
            df_rep4,  # Bloco 5 (180-224)
            df_rep5,  # Bloco 6 (225-269)
            df_rep6,  # Bloco 7 (270-314)
            df_rep7  # Bloco 8 (315-359)
            ], ignore_index=True)

    print(df)
    return df
def interpolate_data(df, resolution=1000):
    phi = np.radians(df['Phi'])
    theta = np.radians(df['Theta'])
    intensity = df['intensityexp']

    phi_grid = np.linspace(np.min(phi), np.max(phi), resolution)
    theta_grid = np.linspace(np.min(theta), np.max(theta), resolution)

    phi_grid, theta_grid = np.meshgrid(phi_grid, theta_grid)

    intensity_grid = griddata((phi, theta), intensity, (phi_grid, theta_grid), method='cubic')

    return phi_grid, theta_grid, intensity_grid

def plot_polar_interpolated(df, resolution=500, line_position=0.5, my_variable=None, save_path=None):
    # Interpolar os dados
    plt.ion()
    phi_grid, theta_grid, intensity_grid = interpolate_data(df, resolution)

    # Criando o gráfico polar
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8), dpi=100)

    # Plotando a intensidade interpolada
    c = ax.pcolormesh(phi_grid, theta_grid, intensity_grid, shading='gouraud', cmap='afmhot')

    # Definir o limite máximo do eixo theta com base no maior valor de theta nos dados
    max_theta = df['Theta'].max()  # Maior valor de theta presente nos dados
    ax.set_ylim(0, np.radians(max_theta))  # Limitar o eixo radial até o maior valor de theta

    # Adiciona rótulos para os ângulos theta, ajustados conforme o máximo de theta nos dados
    theta_ticks = np.linspace(0, max_theta, num=6)  # Definir até 6 ticks no eixo theta
    ax.set_yticks(np.radians(theta_ticks))  # Converte para radianos
    ax.set_yticklabels([f'{int(tick)}°' for tick in theta_ticks])  # Exibe como graus

    # Adicionando a barra de cores
    cbar = fig.colorbar(c, ax=ax, label='', pad=0.08)
    cbar.set_label('', fontsize=22, fontweight='bold')
    cbar.ax.yaxis.set_tick_params(labelsize=30)  # Ajusta o tamanho da fonte
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')  # Deixa os valores em negrito
    # Adiciona a variável fora do gráfico, no canto inferior direito
    if my_variable is not None:
        # Aqui estamos usando coordenadas relativas à figura (0 a 1)
        fig.text(0.82, 0.03, f'{my_variable}', fontsize=34, color='black', ha='right', va='bottom',
                 fontweight='bold')
    Anysotropy = "Anisotropy"
    fig.text(0.94, 0.9,  Anysotropy, fontsize=34, color='black', ha='right', va='bottom',
             fontweight='bold')
    # Definir manualmente os ângulos de Phi (Xticks)
    phi_ticks = np.linspace(0, 2 * np.pi, num=9)[:-1]  # Remove o último valor (360°)
    phi_labels = [f'{int(np.degrees(tick))}°' for tick in phi_ticks]  # Cria os rótulos
    ax.set_xticks(phi_ticks)
    ax.set_xticklabels(phi_labels, fontsize=26, fontweight='bold')

    # Ajustar individualmente os pads de cada rótulo
    pad_values = [1, -1, 3, 0, -7, -6, -1, -6]  # Valores personalizados para cada rótulo
    for label, pad in zip(ax.get_xticklabels(), pad_values):
        label.set_y(label.get_position()[1] + pad * 0.01)  # Move os rótulos individualmente

    # Afasta os rótulos de Phi
    ax.tick_params(pad=8)  # Ajuste global para todos os rótulos
    plt.yticks(fontsize=0, fontweight='bold')
    plt.draw()

    # Salvar a figura, se o caminho de salvamento for fornecido
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Salva no caminho especificado

    # Exibir a figura por 600 segundos
    plt.pause(600)


# Caminho do arquivo
file_path = 'Jonder.txt'
save_path = 'grafico_polar3.png'
df = process_file(file_path)
def rotate_phi_for_plot(df, rotation_angle):
    """
    Mesma rotação, mas garante que Phi = 0 esteja presente **apenas** para plotagem.
    Opera diretamente no df_plot de entrada.
    """
    df['Phi'] = (df['Phi'] + rotation_angle) % 360

    if not np.isclose(df['Phi'].min(), 0):
        df_0 = df[df['Phi'] == df['Phi'].min()].copy()
        df_0['Phi'] = 0
        df = pd.concat([df, df_0], ignore_index=True)

    return df
rotate_angle = 0
df = rotate_phi_for_plot(df, rotate_angle)

r_factor_total="Experimental"
my_variable = r_factor_total
plot_polar_interpolated(df, resolution=500, line_position=0.5, my_variable=my_variable, save_path=save_path)
