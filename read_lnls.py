import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib

matplotlib.use('TkAgg')


# -----------------------------------------------------------------
# FUNÇÕES ESSENCIAIS MANTIDAS
# -----------------------------------------------------------------

def carregar_config(arquivo):
    parametros = {}
    with open(arquivo, 'r') as f:
        for linha in f:
            # Remove comentários (tudo após #)
            linha = linha.split('#', 1)[0].strip()
            if '=' in linha and linha:  # Garante que a linha não está vazia
                chave, valor = linha.split('=', 1)
                chave = chave.strip()
                valor = valor.strip()
                try:
                    # Tenta converter para número (int ou float)
                    if '.' in valor:
                        parametros[chave] = float(valor)
                    else:
                        parametros[chave] = int(valor)
                except ValueError:
                    # Se não for número, mantém como string
                    parametros[chave] = valor
    return parametros


def fourier_symmetrization(theta_values, phi_values, intensity_values, symmetry):
    """
    Aplica a simetrização por expansão em Fourier nos dados XPD.
    (Função mantida exatamente como era)
    """
    n_theta = len(theta_values)
    n_phi = len(phi_values)
    intensity_symmetric = np.zeros_like(intensity_values)

    for i, theta in enumerate(theta_values):
        f = intensity_values[i, :]
        F = np.fft.fft(f)
        F_symmetric = np.zeros_like(F, dtype=complex)
        for u in range(0, n_phi):
            if u % symmetry == 0:
                F_symmetric[u] = F[u]
        f_symmetric = np.fft.ifft(F_symmetric).real
        intensity_symmetric[i, :] = f_symmetric

    return intensity_symmetric


# Função para o polinômio de grau 3 (Usada para calcular o Chi)
def polynomial_3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def process_and_plot(input_file, output_file, plot_dir, phi_values_to_evaluate=None):
    try:
        data = pd.read_csv(input_file, sep='\s+', engine='python')
    except Exception as e:
        print(f"ERRO: Não foi possível ler o arquivo de entrada: {input_file}")
        print(f"Verifique se o arquivo existe e não está vazio. Detalhe: {e}")
        return  # Sai da função se não conseguir ler


    # Agrupa os dados por theta
    grouped = data.groupby('theta')
    results = []

    # Criando valores de phi de acordo com o intervalo definido
    # (phii, phif, dphi vêm do config.txt)
    phi_fine = np.arange(phii, phif + dphi, dphi)

    # Processa cada grupo de theta
    for theta, group in grouped:
        phi = group['phi'].values
        intensity = group['intensity'].values

        # Ajuste polinomial
        try:
            popt, _ = curve_fit(polynomial_3, phi, intensity)
            a, b, c, d = popt
            results.append({'theta': theta, 'a': a, 'b': b, 'c': c, 'd': d})

            # (O resto da lógica de plotagem interna e cálculo de Chi foi mantida)
            intensity_fitted = polynomial_3(phi_fine, *popt)

        except Exception as e:
            print(f"Erro ao ajustar os dados para theta = {theta}: {e}")
            continue

    results_df = pd.DataFrame(results)

    # Salvando os dados ajustados (Chi) no formato esperado
    with open(output_file, 'w') as file:
        num_theta = results_df['theta'].nunique()
        num_phi = len(phi_fine)
        num_points = len(data)

        # Cabeçalho inicial
        file.write(f"      {num_theta}    {num_points}    0     datakind beginning-row linenumbers\n")
        file.write(f"----------------------------------------------------------------\n")
        file.write(f"MSCD Version 1.00 Yufeng Chen and Michel A Van Hove\n")
        file.write(f"Lawrence Berkeley National Laboratory (LBNL), Berkeley, CA 94720\n")
        file.write(f"Copyright (c) Van Hove Group 1997. All rights reserved\n")
        file.write(f"--------------------------------------------------------------\n")
        file.write(f"angle-resolved photoemission extended fine structure (ARPEFS)\n")
        file.write(f"experimental data for Fe 2p3/2 from Fe on STO(100)  excited with hv=1810eV\n")
        file.write(f"\n")
        file.write(f"provided by Pancotti et al. (LNLS in 9, June 2010)\n")
        file.write(f"   intial angular momentum (l) = 1\n")
        file.write(f"   photon polarization angle (polar,azimuth) = (  30.0,   0.0 ) (deg)\n")
        file.write(f"   sample temperature = 300 K\n")
        file.write(f"\n")
        file.write(f"   photoemission angular scan curves\n")
        file.write(f"     (curve point theta phi weightc weighte//k intensity chiexp)\n")
        file.write(f"      {num_theta}     {num_points}       1       {num_theta}     {num_phi}     {num_points}\n")

        # Loop para os diferentes valores de θ
        number_of_theta = 0
        for theta in sorted(results_df['theta'].unique()):
            number_of_theta += 1
            first_row = results_df[results_df['theta'] == theta].iloc[0]
            file.write(
                f"       {number_of_theta}     {num_phi}       19.5900     {first_row['theta']:.4f}      1.00000      0.00000\n"
            )
            subset = results_df[results_df['theta'] == theta]

            group = data[data['theta'] == theta]
            phi = group['phi'].values
            intensity = group['intensity'].values

            phi_fine = np.arange(phii, phif + dphi, dphi)

            for _, row in subset.iterrows():
                # Pega os coeficientes ajustados do polinômio
                a, b, c, d = row['a'], row['b'], row['c'], row['d']

                # --- INÍCIO DA CORREÇÃO ---

                # 1. Calcular a intensidade ajustada NOS PONTOS DE PHI ORIGINAIS (do arquivo)
                #    (Usa 'phi' (14 pts) para o cálculo do Chi)
                intensity_fitted_at_phi = polynomial_3(phi, a, b, c, d)

                # 2. Calcular a intensidade ajustada NO INTERVALO DO CONFIG (para a média)
                #    (Usa 'phi_fine' (12 pts) para a média)
                intensity_fitted_at_phi_fine = polynomial_3(phi_fine, a, b, c, d)

                # --- FIM DA CORREÇÃO ---

                # Calcular a média da intensidade ajustada
                mean_intensity = np.mean(intensity)
                mean_intensity2 = np.mean(intensity_fitted_at_phi_fine)  # Média do ajuste

                # Calcular Chi para cada valor de phi (usando o ajuste nos pontos originais)
                Chi = ((intensity - intensity_fitted_at_phi) / intensity_fitted_at_phi)
                Chi2 = ((intensity - mean_intensity) / mean_intensity)
                Chi3 = ((intensity - mean_intensity2) / mean_intensity2)

            for p, i, chi in zip(phi, intensity, Chi):
                file.write(f"      {p:.5f}      {i:.1f}      {mean_intensity:.1f}      {chi:.7f}\n")
            file.write("")
    print(f"Resultados (Chi) salvos em {output_file}")


# Essa função vai ler o arquivo (coeficientes_ajustados.txt) para a realizar a simetrização
def process_file_fft_input(file_path):
    # (Esta era a sua segunda função 'process_file')
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    theta_value = None

    for i in range(17, len(lines)):  # Começar a ler após o cabeçalho de 17 linhas
        line = lines[i].strip()

        if line:
            parts = line.split()

            if len(parts) == 6:
                theta_value = float(parts[3])  # Lendo o valor de theta

            elif len(parts) == 4 and theta_value is not None:
                phi = float(parts[0])
                col1 = float(parts[1])
                col2 = float(parts[2])
                intensity = float(parts[3])  # Esta é a coluna Chi
                data.append([phi, col1, col2, theta_value, intensity, True])

    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity', 'IsOriginal'])
    print("Valores de Theta lidos para FFT:", df['Theta'].unique())
    return df


def save_to_text_file(data_df, intensity_symmetric, output_file_path):
    """
    Salva os dados atualizados (simetrizados) em um arquivo de texto.
    (Função mantida)
    """
    with open(output_file_path, 'w') as file:
        # Escrever o cabeçalho
        num_theta = len(data_df['Theta'].unique())
        num_points = len(data_df['Phi'].unique()) * num_theta
        num_phi = len(data_df['Phi'].unique())
        file.write(f"      {num_theta}    {num_points}    0     datakind beginning-row linenumbers\n")
        file.write(f"----------------------------------------------------------------\n")
        file.write(f"MSCD Version 1.00 Yufeng Chen and Michel A Van Hove\n")
        file.write(f"Lawrence Berkeley National Laboratory (LBNL), Berkeley, CA 94720\n")
        file.write(f"Copyright (c) Van Hove Group 1997. All rights reserved\n")
        file.write(f"--------------------------------------------------------------\n")
        file.write(f"angle-resolved photoemission extended fine structure (ARPEFS)\n")
        file.write(f"experimental data for Fe 2p3/2 from Fe on STO(100)  excited with hv=1810eV\n")
        file.write(f"\n")
        file.write(f"provided by Pancotti et al. (LNLS in 9, June 2010)\n")
        file.write(f"   intial angular momentum (l) = 1\n")
        file.write(f"   photon polarization angle (polar,azimuth) = (  30.0,   0.0 ) (deg)\n")
        file.write(f"   sample temperature = 300 K\n")
        file.write(f"\n")
        file.write(f"   photoemission angular scan curves\n")
        file.write(f"     (curve point theta phi weightc weighte//k intensity chiexp)\n")
        file.write(f"      {num_theta}     {num_points}       1       {num_theta}     {num_phi}     {num_points}\n")

        number_of_theta = 0
        for theta in sorted(data_df['Theta'].unique()):
            number_of_theta += 1
            first_row = data_df[data_df['Theta'] == theta].iloc[0]
            file.write(
                f"       {number_of_theta}     {num_phi}       19.5900     {first_row['Theta']:.4f}      1.00000      0.00000\n"
            )
            theta_data = data_df[data_df['Theta'] == theta]
            sorted_phi_indices = theta_data['Phi'].argsort()
            for j, phi in enumerate(theta_data['Phi'].values[sorted_phi_indices]):
                col1 = theta_data.iloc[j]['Col1']
                col2 = theta_data.iloc[j]['Col2']
                intensity = intensity_symmetric[number_of_theta - 1, j]  # Usa o valor simetrizado
                file.write(f"      {phi:.5f}      {col1:.1f}      {col2:.1f}      {intensity:.7f}\n")
                file.write("")


# (Esta era a sua terceira função 'process_file', agora renomeada para clareza)
# (Esta era a sua terceira função 'process_file', agora renomeada para clareza)
def process_file_for_plot(file_path, sigma=3, rotate_angle=0):
    # (Função mantida)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    theta_value = None

    for i in range(17, len(lines)):
        line = lines[i].strip()
        if line:
            parts = line.split()
            if len(parts) == 6:
                theta_value = float(parts[3])
            elif len(parts) == 4 and theta_value is not None:
                phi = float(parts[0])
                col1 = float(parts[1])
                col2 = float(parts[2])
                intensity = float(parts[3])
                data.append([phi, col1, col2, theta_value, intensity, True])

    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity', 'IsOriginal'])

    # Suavização
    df['Smoothed_Intensity'] = np.nan
    for theta_value in df['Theta'].unique():
        df_theta = df[df['Theta'] == theta_value].sort_values(by='Phi').copy()
        if len(df_theta) > 1:
            smoothed_intensity = gaussian_filter(df_theta['Intensity'].values, sigma=sigma)
            df.loc[df['Theta'] == theta_value, 'Smoothed_Intensity'] = smoothed_intensity
        elif len(df_theta) == 1:
            df.loc[df['Theta'] == theta_value, 'Smoothed_Intensity'] = df_theta['Intensity'].values[0]

    # (Lógica de rotação e replicação para plotagem mantida)

    ####################################################################################################################
    df_plot = df.copy()

    def rotate_phi_for_plot(df_plot, rotation_angle):
        df_plot['Phi'] = (df_plot['Phi'] + rotation_angle) % 360
        if not np.isclose(df_plot['Phi'].min(), 0):
            df_0 = df_plot[df_plot['Phi'] == df_plot['Phi'].min()].copy()
            df_0['Phi'] = 0
            df_plot = pd.concat([df_plot, df_0], ignore_index=True)
        return df_plot

    df_plot = rotate_phi_for_plot(df_plot, rotate_angle)
    phi_min2 = df_plot['Phi'].min()
    phi_max2 = df_plot['Phi'].max()
    phi_interval2 = phi_max2 - phi_min2

    # Adicionar ponto 360
    if phi_interval2 < 360:
        phi_min2 = df_plot['Phi'].min()
        df_360 = df_plot[np.isclose(df_plot['Phi'], phi_min2)].copy()
        df_360['Phi'] = 360
        df_plot = pd.concat([df_plot, df_360], ignore_index=True)

    # Lógica de Replicação
    if phi_interval2 == 120:
        # (Sua lógica C3 original)
        first_values = df_plot.groupby('Theta').first().reset_index()
        df_plot = df_plot.groupby('Theta', group_keys=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df_plot.groupby('Theta').last().reset_index()
        last_values['Phi'] = first_values['Phi']
        df_plot = pd.concat([df_plot, last_values], ignore_index=True)
        df_0_120 = df_plot.copy()
        df_0_120['Phi'] = 120 + df_0_120['Phi']
        df_0_120['isOriginal'] = False
        df_240_360 = df_plot.copy()
        df_240_360['Phi'] = 240 + df_240_360['Phi']
        df_plot = pd.concat([df_plot, df_0_120, df_240_360]).reset_index(drop=True)
    if phi_interval2 == 90:
        # (Sua lógica C4 original)
        first_values = df_plot.groupby('Theta').first().reset_index()
        df_plot = df_plot.groupby('Theta', group_keys=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df_plot.groupby('Theta').last().reset_index()
        last_values['Phi'] = first_values['Phi']
        df_plot = pd.concat([df_plot, last_values], ignore_index=True)
        df_0_90 = df_plot.copy()
        df_0_90['Phi'] = 90 + df_0_90['Phi']
        df_90_180 = df_plot.copy()
        df_90_180['Phi'] = 180 + df_90_180['Phi']
        df_180_270 = pd.concat([df_plot, df_0_90]).reset_index(drop=True)
        df_180_270['Phi'] = 180 + df_180_270['Phi']
        df_plot = pd.concat([df_plot, df_0_90, df_180_270]).reset_index(drop=True)

    ### NOVO BLOCO ###
    # Lógica para C8 (8-fold / 45 graus).
    # Usamos np.isclose para o seu intervalo de 44 graus (120 - 76).
    if np.isclose(phi_interval2, 44):
        print(f"Detectado intervalo de {phi_interval2} graus. Aplicando replicação C8 (8-fold).")

        # (Sua lógica de fechar o loop, pegando o primeiro e o último ponto)
        first_values = df_plot.groupby('Theta').first().reset_index()
        df_plot = df_plot.groupby('Theta', group_keys=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df_plot.groupby('Theta').last().reset_index()
        last_values['Phi'] = first_values['Phi']
        df_plot = pd.concat([df_plot, last_values], ignore_index=True)

        # Criar os 7 blocos replicados
        df_rep1 = df_plot.copy();
        df_rep1['Phi'] = 45 + df_rep1['Phi']
        df_rep2 = df_plot.copy();
        df_rep2['Phi'] = 90 + df_rep2['Phi']
        df_rep3 = df_plot.copy();
        df_rep3['Phi'] = 135 + df_rep3['Phi']
        df_rep4 = df_plot.copy();
        df_rep4['Phi'] = 180 + df_rep4['Phi']
        df_rep5 = df_plot.copy();
        df_rep5['Phi'] = 225 + df_rep5['Phi']
        df_rep6 = df_plot.copy();
        df_rep6['Phi'] = 270 + df_rep6['Phi']
        df_rep7 = df_plot.copy();
        df_rep7['Phi'] = 315 + df_rep7['Phi']

        # Juntar tudo
        df_plot = pd.concat([
            df_plot,
            df_rep1, df_rep2, df_rep3, df_rep4, df_rep5, df_rep6, df_rep7
        ]).reset_index(drop=True)

    ### FIM DO NOVO BLOCO ###

    if phi_interval2 == 177:
        # (Sua lógica C2 original)
        first_values = df_plot.groupby('Theta').first().reset_index()
        df_plot = df_plot.groupby('Theta', group_keys=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df.groupby('Theta').last().reset_index()
        last_values['Phi'] = first_values['Phi']
        df_plot = pd.concat([df_plot, last_values], ignore_index=True)
        df_0_180 = df_plot.copy()
        df_0_180['Phi'] = 180 + df_0_180['Phi']
        df_plot = pd.concat([df_plot, df_0_180]).reset_index(drop=True)

    ####################################################################################################################

    # (O restante da função de lógica de rotação para o 'df' principal)

    def rotate_phi(df, rotation_angle):
        df['Phi'] = (df['Phi'] + rotation_angle) % 360
        return df

    df = rotate_phi(df, rotate_angle)
    phi_min = df['Phi'].min()
    phi_max = df['Phi'].max()
    phi_interval = phi_max - phi_min
    if phi_interval < 360:
        phi_min = df['Phi'].min()
        df_360 = df[np.isclose(df['Phi'], phi_min)].copy()
        df_360['Phi'] = 360
        df = pd.concat([df, df_360], ignore_index=True)

    # (A lógica de replicação para 'df' também precisa ser atualizada)
    if phi_interval == 120:
        # (Sua lógica C3 original)
        first_values = df.groupby('Theta').first().reset_index()
        df = df.groupby('Theta', group_keys=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df.groupby('Theta').last().reset_index()
        last_values['Phi'] = first_values['Phi']
        df = pd.concat([df, last_values], ignore_index=True)
        df_0_120 = df.copy()
        df_0_120['Phi'] = 120 + df_0_120['Phi']
        df_0_120['isOriginal'] = False
        df_240_360 = df.copy()
        df_240_360['Phi'] = 240 + df_240_360['Phi']
        df = pd.concat([df, df_0_120, df_240_360]).reset_index(drop=True)
    if phi_interval == 90:
        # (Sua lógica C4 original)
        first_values = df.groupby('Theta').first().reset_index()
        df = df.groupby('Theta', group_keys=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df.groupby('Theta').last().reset_index()
        last_values['Phi'] = first_values['Phi']
        df = pd.concat([df, last_values], ignore_index=True)
        df_0_90 = df.copy()
        df_0_90['Phi'] = 90 + df_0_90['Phi']
        df_90_180 = df.copy()
        df_90_180['Phi'] = 180 + df_90_180['Phi']
        df_180_270 = pd.concat([df, df_0_90]).reset_index(drop=True)
        df_180_270['Phi'] = 180 + df_180_270['Phi']
        df = pd.concat([df, df_0_90, df_180_270]).reset_index(drop=True)

    ### NOVO BLOCO (para 'df') ###
    if np.isclose(phi_interval, 44):
        # (Lógica para fechar o loop)
        first_values = df.groupby('Theta').first().reset_index()
        df = df.groupby('Theta', group_keys=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df.groupby('Theta').last().reset_index()
        last_values['Phi'] = first_values['Phi']
        df = pd.concat([df, last_values], ignore_index=True)

        # Marcar os blocos replicados como "não originais"
        df_rep1 = df.copy();
        df_rep1['Phi'] = 45 + df_rep1['Phi'];
        df_rep1['IsOriginal'] = False
        df_rep2 = df.copy();
        df_rep2['Phi'] = 90 + df_rep2['Phi'];
        df_rep2['IsOriginal'] = False
        df_rep3 = df.copy();
        df_rep3['Phi'] = 135 + df_rep3['Phi'];
        df_rep3['IsOriginal'] = False
        df_rep4 = df.copy();
        df_rep4['Phi'] = 180 + df_rep4['Phi'];
        df_rep4['IsOriginal'] = False
        df_rep5 = df.copy();
        df_rep5['Phi'] = 225 + df_rep5['Phi'];
        df_rep5['IsOriginal'] = False
        df_rep6 = df.copy();
        df_rep6['Phi'] = 270 + df_rep6['Phi'];
        df_rep6['IsOriginal'] = False
        df_rep7 = df.copy();
        df_rep7['Phi'] = 315 + df_rep7['Phi'];
        df_rep7['IsOriginal'] = False

        df = pd.concat([
            df,
            df_rep1, df_rep2, df_rep3, df_rep4, df_rep5, df_rep6, df_rep7
        ]).reset_index(drop=True)
    ### FIM DO NOVO BLOCO ###

    if phi_interval == 177:
        # (Sua lógica C2 original)
        first_values = df.groupby('Theta').first().reset_index()
        df = df.groupby('Theta', group_keys=False).apply(lambda x: x.drop(x.index[0]))
        last_values = df.groupby('Theta').last().reset_index()
        last_values['Phi'] = first_values['Phi']
        df = pd.concat([df, last_values], ignore_index=True)
        df_0_180 = df.copy()
        df_0_180['Phi'] = 180 + df_0_180['Phi']
        df = pd.concat([df, df_0_180]).reset_index(drop=True)

    return df, df_plot

# Função para interpolar os dados
def interpolate_data(df, resolution=1000):
    # (Função mantida)
    phi = np.radians(df['Phi'])
    theta = np.radians(df['Theta'])
    intensity = df['Smoothed_Intensity']  # Usar a intensidade suavizada
    phi_grid = np.linspace(np.min(phi), np.max(phi), resolution)
    theta_grid = np.linspace(np.min(theta), np.max(theta), resolution)
    phi_grid, theta_grid = np.meshgrid(phi_grid, theta_grid)
    intensity_grid = griddata((phi, theta), intensity, (phi_grid, theta_grid), method='cubic')
    return phi_grid, theta_grid, intensity_grid


# Função para gerar o gráfico polar
def plot_polar_interpolated(df, resolution=500, line_position=0.5, my_variable='Experimental', save_path=None):
    # (Função mantida)
    plt.ion()
    phi_grid, theta_grid, intensity_grid = interpolate_data(df, resolution)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8), dpi=100)
    c = ax.pcolormesh(phi_grid, theta_grid, intensity_grid, shading='gouraud', cmap='afmhot')
    max_theta = df['Theta'].max()
    ax.set_ylim(0, np.radians(max_theta))
    theta_ticks = np.linspace(0, max_theta, num=6)
    ax.set_yticks(np.radians(theta_ticks))
    ax.set_yticklabels([f'{int(tick)}°' for tick in theta_ticks])
    cbar = fig.colorbar(c, ax=ax, label='', pad=0.08)
    cbar.set_label('', fontsize=22, fontweight='bold')
    cbar.ax.yaxis.set_tick_params(labelsize=30)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    if my_variable is not None:
        fig.text(0.82, 0.03, f'{my_variable}', fontsize=34, color='black', ha='right', va='bottom',
                 fontweight='bold')
    Anysotropy = "Anisotropy"
    fig.text(0.94, 0.9, Anysotropy, fontsize=34, color='black', ha='right', va='bottom',
             fontweight='bold')
    phi_ticks = np.linspace(0, 2 * np.pi, num=9)[:-1]
    phi_labels = [f'{int(np.degrees(tick))}°' for tick in phi_ticks]
    ax.set_xticks(phi_ticks)
    ax.set_xticklabels(phi_labels, fontsize=26, fontweight='bold')
    pad_values = [1, -1, 3, 0, -7, -6, -1, -6]
    for label, pad in zip(ax.get_xticklabels(), pad_values):
        label.set_y(label.get_position()[1] + pad * 0.01)
    ax.tick_params(pad=8)
    plt.yticks(fontsize=0, fontweight='bold')
    plt.draw()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.pause(600)


def save_to_txt_with_blocks(df, file_name):
    """
    Salva os dados organizados em blocos (para o 'arquivo_saida' do config)
    (Função mantida)
    """
    df_original = df[df['IsOriginal']]
    num_theta = df_original['Theta'].nunique()
    num_phi = df_original['Phi'].nunique() - 1
    num_points = num_theta * num_phi

    with open(file_name, 'w', newline="\n") as file:
        file.write(f"   323    17    0     datakind beginning-row linenumbers\n")
        file.write(f"----------------------------------------------------------------\n")
        file.write(f"MSCD Version 1.00 Yufeng Chen and Michel A Van Hove\n")
        file.write(f"Lawrence Berkeley National Laboratory (LBNL), Berkeley, CA 94720\n")
        file.write(f"Copyright (c) Van Hove Group 1997. All rights reserved\n")
        file.write(f"--------------------------------------------------------------\n")
        file.write(f" angle-resolved photoemission extended fine structure (ARPEFS)\n")
        file.write(f" experimental data for Pd 3d3/2 from W(100)  excited with hv=1810eV\n")
        file.write("\n")
        file.write(f" provided by Pancotti et al. (LNLS in 12, June 2007)\n")
        file.write(f"   intial angular momentum (l) = 2\n")
        file.write(f"   photon polarization angle (polar,azimuth) = (  30.0,   0.0 ) (deg)\n")
        file.write(f"   sample temperature = 300 K\n")
        file.write("\n")
        file.write(f"   photoemission angular scan curves\n")
        file.write(f"     (curve point theta phi weightc weighte//k intensity chiexp)\n")
        file.write(f"      {num_theta}    {num_points}       1      {num_theta}     {num_phi}    {num_points}\n")

        number_of_theta = 0
        for theta in sorted(df_original['Theta'].unique()):
            number_of_theta += 1
            subset = df_original[df_original['Theta'] == theta].sort_values(by='Phi')
            first_intensity = subset.iloc[0]['Intensity']
            last_intensity = subset.iloc[-1]['Intensity']
            if number_of_theta < 10:
                theta_format = f"       {number_of_theta}"
            else:
                theta_format = f"      {number_of_theta}"
            file.write(
                f"{theta_format}     {num_phi}      9.85000      {theta:1.0f}      1.00000      0.00000\n")

            for i, row in enumerate(subset.itertuples(index=False)):
                if not (i == len(subset) - 1 and row.Intensity == first_intensity):
                    if row.Phi < 10:
                        phi_format = f"{row.Phi:7.5f}"
                    elif row.Phi < 100:
                        phi_format = f"{row.Phi:7.4f}"
                    else:
                        phi_format = f"{row.Phi:7.3f}"
                    if row.Col1 < 100000:
                        col1_format = f"{row.Col1:1.1f}"
                    else:
                        col1_format = f"{row.Col1:1.0f}."
                    if row.Col2 < 100000:
                        col2_format = f"{row.Col2:1.1f}"
                    else:
                        col2_format = f"{row.Col2:1.0f}."
                    file.write(f"      {phi_format}      {col1_format}      {col2_format}   {row.Intensity: 10.7f}\n")


# -----------------------------------------------------------------
# CORPO PRINCIPAL DO SCRIPT (MODIFICADO)
# -----------------------------------------------------------------

# 1. Carregar parâmetros do config.txt
#    (Note que parâmetros como 'file_prefix', 'channel', 'thetai'
#     não são mais usados, mas não causam mal)
parametros = carregar_config('config.txt')

phii = parametros["phii"]
phif = parametros["phif"]
dphi = parametros["dphi"]
symmetry = parametros["symmetry"]
arquivo_saida = parametros["arquivo_saida"]
rotate_angle = parametros["rotate_angle"]

# Caminho para o arquivo de saída final (pós-FFT)
output_file_path_fft = '../simetrizados.txt'

# -----------------------------------------------------------------
# ETAPA 1: Calcular Chi (Normalização)
# -----------------------------------------------------------------
print("--- ETAPA 1: Iniciando cálculo do Chi ---")

# ATENÇÃO: Definimos o arquivo de entrada manualmente
#          para usar o C1 que acabamos de gerar.
#          Mude para "C2", "C3" ou "C4" se desejar.
input_file = "../saidatpintensity_C1.txt"

# Arquivo de saída desta etapa (entrada para a próxima)
output_file_chi = "../coeficientes_ajustados.txt"
plot_dir = "../plots"

os.makedirs(plot_dir, exist_ok=True)

# Define o range de phi para o ajuste polinomial
phi_values = np.arange(phii, phif + dphi, dphi)

# Executa o cálculo do Chi e salva em 'coeficientes_ajustados.txt'
process_and_plot(input_file, output_file_chi, plot_dir, phi_values)

# -----------------------------------------------------------------
# ETAPA 2: Aplicar Simetrização (FFT)
# -----------------------------------------------------------------
print("--- ETAPA 2: Iniciando Simetrização (FFT) ---")

# Leitura dos dados (do arquivo de Chi)
data_df = process_file_fft_input(output_file_chi)

# Organizar os dados para a simetrização
theta_values = np.sort(data_df['Theta'].unique())
phi_values_fft = data_df['Phi'].unique()  # Nome diferente para evitar conflito
n_theta = len(theta_values)
n_phi = len(phi_values_fft)

# Criar uma matriz de intensidades (Chi)
intensity_values = np.zeros((n_theta, n_phi))
for i, theta in enumerate(theta_values):
    theta_data = data_df[data_df['Theta'] == theta]
    # Garantir que os Phis estão ordenados
    intensity_values[i, :] = theta_data.sort_values(by='Phi')['Intensity'].values

# Aplicar simetrização
intensity_symmetric = fourier_symmetrization(theta_values, phi_values_fft, intensity_values, symmetry)

# Salvar os resultados simetrizados em 'simetrizados.txt'
save_to_text_file(data_df, intensity_symmetric, output_file_path_fft)
print(f"Dados simetrizados salvos em: {output_file_path_fft}")

# -----------------------------------------------------------------
# ETAPA 3: Plotar o resultado final e Salvar (se necessário)
# -----------------------------------------------------------------
print("--- ETAPA 3: Iniciando Plotagem e Salvamento Final ---")

# Processar os dados de 'simetrizados.txt' para plotagem e salvamento
df, df_plot = process_file_for_plot(output_file_path_fft, 1, rotate_angle)

# Plotar o gráfico polar
plot_polar_interpolated(df_plot)

# Salvar o arquivo final no formato 'arquivo_saida' (ex: expGarotate.txt)
save_to_txt_with_blocks(df, arquivo_saida)
print(f"Arquivo final salvo em: {arquivo_saida}")

print("--- Processamento Concluído ---")