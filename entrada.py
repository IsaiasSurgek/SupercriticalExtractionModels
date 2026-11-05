"""ENTRADA DE DADOS — edite este arquivo e mais nenhum.

Aqui vão: a matéria-prima, o equipamento e as curvas cinéticas.

Unidades de entrada
-------------------
tempo             minutos
massa recolhida   gramas
massa seca        gramas
pressão           bar
temperatura       °C
vazão             g/min
densidades        kg/m3
"""

from __future__ import annotations

# =============================================================================
# 1. MATÉRIA-PRIMA
# =============================================================================
# Troque este bloco inteiro ao mudar de planta.

MATERIA_PRIMA = "Alecrim (Rosmarinus officinalis), Mesh 28"

TEOR_OLEO = 0.13803
"""Fração extraível da matriz seca [g óleo / g matriz seca].

É o rendimento do Soxhlet. Define três coisas de uma vez:
  - x0, o teor inicial em base livre de soluto = TEOR/(1-TEOR)
  - N, a carga de sólido inerte = massa_seca * (1 - TEOR)
  - o teto físico de e_lim no modelo de Esquível

Se você não tem esse número para a sua planta, o ajuste inteiro fica sem
referência de escala. Meça antes.
"""

RHO_PARTICULA = 1046.0
"""Densidade real da partícula [kg/m3]. Entra só na porosidade do leito."""

DIAMETRO_PARTICULA_MM = 0.595
"""Diâmetro médio da partícula [mm]. Usado apenas para relatar a área
específica estimada; não entra no ajuste."""


# =============================================================================
# 2. EQUIPAMENTO
# =============================================================================

VOLUME_EXTRATOR_ML = 42.0
"""Volume útil do extrator [mL]."""

VOLUME_INERTES_ML = 0.2827
"""Volume de esferas de vidro ou outro inerte no leito [mL].
Zero se não houver. (20 esferas de 3 mm = 0,2827 mL.)"""

# --- vazão medida por bolhômetro (recomendado) -------------------------------
VAZAO_BOLHOMETRO_ML_MIN = 6.04
"""Leitura do bolhômetro [mL de CO2 GASOSO / min], à pressão e temperatura
ambientes, na saída da unidade.

Deixe ``None`` se você não mediu assim e prefere informar VAZAO_G_MIN direto.

O bolhômetro mede volume de gás despressurizado, então a conversão para massa
usa a densidade do CO2 gasoso no ambiente (~1,8 g/L), e não a densidade
supercrítica dentro do extrator (200 a 960 kg/m3). Confundir as duas erra por
duas a três ordens de grandeza — foi exatamente o que aconteceu nos trabalhos
anteriores deste grupo, que usaram 1 g/L e inflaram a solubilidade em ~295x.
"""

T_AMBIENTE_C = 25.0
"""Temperatura ambiente na saída do bolhômetro [°C]. Entre 20 e 25 °C a vazão
mássica muda 1,7 %."""

VAZAO_G_MIN = None
"""Vazão mássica de CO2 [g/min], se você já a tem em massa.

Tem precedência sobre VAZAO_BOLHOMETRO_ML_MIN. Deixe ``None`` para usar o
bolhômetro; se ambos forem ``None``, cai em VAZAO_ESPECIFICA * massa_seca.

ATENÇÃO: o que os dados determinam é o produto Q*y_r, não Q e y_r separados.
Com a vazão medida, y_r fica determinado; sem ela, o y_r estimado é aparente.
"""

VAZAO_ESPECIFICA_G_G_MIN = 0.1806
"""Vazão específica de literatura [g CO2 / (g matriz . min)].

Último recurso, usado só se VAZAO_BOLHOMETRO_ML_MIN e VAZAO_G_MIN forem ambos
``None``. Média geométrica de oito referências de alecrim (DOMINGOS, 2025),
cuja faixa vai de 0,0126 a 1,066 — 84x de amplitude. Preferir sempre a medida.
"""


# =============================================================================
# 3. DENSIDADE DO CO2
# =============================================================================

FONTE_DENSIDADE = "coolprop"
"""De onde vêm as densidades do CO2. Três opções:

``"tabela"``
    Usa só a tabela DENSIDADE_CO2 abaixo. Ignora o CoolProp mesmo que esteja
    instalado. Escolha esta para garantir que o resultado não muda se a versão
    do CoolProp mudar, ou para reproduzir um cálculo antigo.

``"coolprop"``
    Usa só o CoolProp (equação de Span-Wagner), para qualquer P e T. Ignora a
    tabela. Exige ``pip install CoolProp``.

``"auto"``  (padrão)
    Tabela primeiro; o que faltar, pede ao CoolProp. Se o CoolProp não estiver
    instalado e faltar alguma condição, o programa para e explica.

Nota: densidade do CO2 é propriedade do fluido — **não muda quando você troca
de matéria-prima**. A tabela abaixo continua válida para qualquer planta; só
acrescente as condições novas se usar outras pressões e temperaturas.
"""

# Uma entrada por condição: (pressão em bar, temperatura em °C): densidade kg/m3
# Consulte NIST WebBook (webbook.nist.gov/chemistry/fluid) ou CoolProp.
# Pode ficar vazia ({}) se FONTE_DENSIDADE = "coolprop".

DENSIDADE_CO2 = {
    (90, 60): 235.39,
    (100, 40): 628.61,
    (100, 50): 384.33,
    (200, 50): 784.29,
    (300, 60): 829.71,
}


# =============================================================================
# 4. CURVAS CINÉTICAS
# =============================================================================
# Uma entrada por corrida. Acrescente quantas quiser.
#
#   "P_bar"        pressão [bar]
#   "T_C"          temperatura [°C]
#   "massa_seca_g" massa seca de matriz carregada no extrator [g]
#   "tempo_min"    tempos de coleta [min]
#   "massa_g"      massa recolhida em cada tempo [g]  -- ver MASSA_ACUMULADA
#
# As listas "tempo_min" e "massa_g" precisam ter o mesmo tamanho.

MASSA_ACUMULADA = True
"""True  = "massa_g" é a massa TOTAL recolhida até aquele tempo.
False = "massa_g" é a massa recolhida NAQUELE intervalo (o programa soma)."""

CURVAS = {
    # -------------------------------------------------------------------------
    # Fonte: planilha "Extrações realizadas.xlsx", uma aba por corrida
    # ("Supercrítico <data>"), bloco "Coletas". A massa seca é a "Pesagem 2"
    # (depois da secagem a 60 °C por 30 min).
    #
    #   E1  12/03/2025    90 bar / 60 °C
    #   E2  12/09/2024   100 bar / 40 °C  ) duplicata
    #   E3  21/11/2024   100 bar / 40 °C  )
    #   E4  25/02/2025   100 bar / 50 °C
    #   E5  26/09/2024   200 bar / 50 °C  ) duplicata
    #   E6  07/11/2024   200 bar / 50 °C  )
    #   E7  30/09/2024   300 bar / 60 °C
    #
    # Dado bruto da planilha, menos quatro leituras descartadas. Em três casos
    # (E1 75 min, E4 135 min, E6 165 min) a leitura é alta: vem seguida de
    # massa acumulada MENOR, o que é impossível numa curva acumulada, e é
    # também o maior desvio positivo contra os dois modelos. No quarto caso
    # (E6 120 min) a leitura é baixa, abaixo da do ponto anterior.
    # Cada descarte está anotado na curva correspondente, com o valor.
    # -------------------------------------------------------------------------

    # --- 90 bar / 60 °C ------------------------------------------------------
    "E1": {                               # 12/03/2025, 165 min
        # Fora: 75 min (0,0051 g) — leitura alta, +18,9 % acima do modelo, e
        # seguida pelos 0,0046 g de 90 min.
        "P_bar": 90,
        "T_C": 60,
        "massa_seca_g": 9.7784,
        "tempo_min": [15, 30, 45, 60, 90, 105, 120, 135, 150, 165],
        "massa_g": [0.0018, 0.0026, 0.0033, 0.0036, 0.0046,
                    0.0050, 0.0056, 0.0060, 0.0062, 0.0064],
    },

    # --- 100 bar / 40 °C -----------------------------------------------------
    "E2": {                               # 12/09/2024, 180 min
        "P_bar": 100,
        "T_C": 40,
        "massa_seca_g": 9.6460,
        "tempo_min": [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180],
        "massa_g": [0.0023, 0.0033, 0.0033, 0.0036, 0.0055, 0.0088,
                    0.0110, 0.0149, 0.0172, 0.0178, 0.0247, 0.0252],
    },
    "E3": {                               # 21/11/2024, 300 min
        "P_bar": 100,
        "T_C": 40,
        "massa_seca_g": 9.7659,
        "tempo_min": [15, 30, 45, 60, 75, 90, 105, 120, 135, 150,
                      165, 180, 195, 210, 225, 240, 255, 270, 285, 300],
        "massa_g": [0.0022, 0.0042, 0.0049, 0.0054, 0.0054, 0.0087, 0.0116, 0.0123, 0.0174, 0.0208,
                    0.0255, 0.0263, 0.0267, 0.0291, 0.0329, 0.0346, 0.0374, 0.0385, 0.0396, 0.0416],
    },

    # --- 100 bar / 50 °C -----------------------------------------------------
    "E4": {                               # 25/02/2025, 195 min
        # Fora: 135 min (0,0127 g) — leitura alta, +13,8 % acima do modelo, e
        # seguida pelos 0,0124 g de 150 min.
        "P_bar": 100,
        "T_C": 50,
        "massa_seca_g": 9.6834,
        "tempo_min": [15, 30, 45, 60, 75, 90, 105, 120, 150, 165, 180, 195],
        "massa_g": [0.0029, 0.0033, 0.0042, 0.0060, 0.0065, 0.0073,
                    0.0095, 0.0109, 0.0124, 0.0138, 0.0138, 0.0138],
    },

    # --- 200 bar / 50 °C -----------------------------------------------------
    "E5": {                               # 26/09/2024, 225 min
        "P_bar": 200,
        "T_C": 50,
        "massa_seca_g": 9.7134,
        "tempo_min": [15, 30, 45, 60, 75, 90, 105, 120,
                      135, 150, 165, 180, 195, 210, 225],
        "massa_g": [0.0009, 0.0034, 0.0076, 0.0162, 0.0183, 0.0268, 0.0290, 0.0335,
                    0.0383, 0.0383, 0.0415, 0.0447, 0.0475, 0.0495, 0.0503],
    },
    "E6": {                               # 07/11/2024, 255 min
        # Fora: 165 min (0,0389 g) — leitura alta, +22,6 % acima do modelo, e
        # seguida pelos 0,0386 g de 180 min. Fora também 120 min (0,0099 g),
        # abaixo dos 0,0103 g de 105 min.
        "P_bar": 200,
        "T_C": 50,
        "massa_seca_g": 9.8471,
        "tempo_min": [15, 30, 45, 60, 75, 90, 105, 135, 150, 180,
                      195, 210, 225, 240, 255],
        "massa_g": [0.0024, 0.0038, 0.0051, 0.0072, 0.0079, 0.0096, 0.0103, 0.0182, 0.0261, 0.0386,
                    0.0422, 0.0452, 0.0477, 0.0501, 0.0610],
    },

    # --- 300 bar / 60 °C -----------------------------------------------------
    "E7": {                               # 30/09/2024, 300 min
        "P_bar": 300,
        "T_C": 60,
        "massa_seca_g": 9.6086,
        "tempo_min": [15, 30, 45, 60, 75, 90, 105, 120, 135, 150,
                      165, 180, 195, 210, 225, 240, 255, 270, 285, 300],
        "massa_g": [0.0035, 0.0077, 0.0091, 0.0157, 0.0176, 0.0207, 0.0245, 0.0309, 0.0356, 0.0385,
                    0.0454, 0.0508, 0.0528, 0.0559, 0.0588, 0.0620, 0.0646, 0.0654, 0.0662, 0.0688],
    },
}


# =============================================================================
# 5. OPÇÕES DO AJUSTE
# =============================================================================

MODO_YR = "fitted"
"""Como tratar a solubilidade y_r:
  "fitted"  y_r estimado junto com o resto, uma por condição (recomendado)
  "legacy"  y_r calculado do primeiro ponto: m1/(Q*t1)
  "fixed"   y_r imposto por você, em YR_FIXO_G_KG
"""

YR_FIXO_G_KG = 2.0
"""y_r usado quando MODO_YR == "fixed" [g óleo / kg CO2]."""

AJUSTE_GLOBAL_SA = True
"""Se True, faz também um ajuste com um único ``sa`` para todas as curvas.
``sa`` é propriedade da moagem: mesmo lote e mesma peneira devem dar o mesmo
valor, então deixá-lo variar entre condições costuma ser artefato de ajuste."""

N_PARTIDAS = 64
"""Partidas do multi-start por curva. Abaixo de ~50 o otimizador começa a
perder o ótimo global em superfícies com vários mínimos locais."""

CALCULAR_SOLUBILIDADE = True
"""Liga ou desliga os modelos de solubilidade (Chrastil e del Valle & Aguilera).

Não exige nenhum dado além dos que já estão neste arquivo: a solubilidade sai do
último ponto de cada curva, ``S = massa de extrato / massa de CO2``, com a
densidade do CO2 vinda de FONTE_DENSIDADE.

Vale desligar quando:

* você quer só a cinética e está iterando rápido;
* a premissa de saturação não se sustenta nas suas curvas. O módulo testa isso
  e avisa: se ``m_extrato`` contra ``m_CO2`` não for reta pela origem, o leito
  empobreceu antes do fim da corrida e o último ponto **subestima** a
  solubilidade. Nesse caso o número sai, mas não é solubilidade de saturação.

A linha de comando tem precedência: ``python main.py --sem-solubilidade``
desliga mesmo com esta opção em True.
"""

PASTA_SAIDA = "resultados"
"""Pasta onde tudo é gravado.

Caminho relativo é resolvido **a partir da pasta da ferramenta**, não do
diretório de onde você chamou o script. Assim `python ferramenta/main.py`
rodado de qualquer lugar deixa os resultados em `ferramenta/resultados/`,
junto do código.

Para gravar em outro lugar, informe um caminho absoluto, por exemplo
``r"D:\Doc\Faculdade\resultados_alecrim"``.
"""
