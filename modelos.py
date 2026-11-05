"""Modelos cinéticos: Sovová (1994) e Esquível et al. (1999).

SOVOVÁ (1994) — implementação numericamente estável.

Convenção adotada
-----------------
``x_k`` é o soluto em **células intactas** (difícil acesso, extraído pelo
mecanismo interno lento) e ``x0 - x_k`` é o de **células rompidas** (livre,
extraído no regime de solubilidade). Define-se ``sa = x_k / x0``.

Grupos adimensionais::

    q   = (Q/N) * t                          kg CO2 / kg inerte
    Z   = Kfa0 * rho_CO2 * N / ((1-eps) * rho_s * Q)
    W   = Ksa0 * N / ((1-eps) * Q)
    q_m = (x0 - x_k) / (y_r * Z)
    q_n = q_m + (1/W) * ln[(x_k + (x0-x_k)*exp(W*x0/y_r)) / x0]

Três períodos::

    q <  q_m          e = q * y_r * (1 - exp(-Z))
    q_m <= q < q_n    e = y_r * [q - q_m * exp(z_k - Z)]
    q >= q_n          e = x0 - (y_r/W) * ln{1 + [exp(W*x0/y_r) - 1] *
                                              exp(W(q_m - q)) * (x_k/x0)}

Por que a implementação é em espaço logarítmico
-----------------------------------------------
``exp(W*x0/y_r)`` estoura em ponto flutuante para combinações inteiramente
plausíveis: ``A = W*x0/y_r`` chega a 10^4 em dados reais de extração
incompleta. Avaliado ingenuamente, o termo vira ``inf``, ``q_n`` vira ``inf``,
o terceiro período fica inalcançável **sem nenhum aviso**, e o modelo passa a
ser outro modelo.

Aqui os três períodos usam ``log1p``, ``expm1`` e ``logaddexp``. Nenhuma
expressão intermediária sai da faixa de ponto flutuante dentro dos limites
físicos dos parâmetros.

Unidades (SI)
-------------
t [s], Q [kg/s], N [kg], densidades [kg/m3], Kfa0 e Ksa0 [1/s],
x0, e, y_r em fração (kg/kg) — nunca porcentagem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SA_MIN = 1e-6
SA_MAX = 1.0 - 1e-6

__all__ = [
    "GruposSovova",
    "grupos_sovova",
    "sovova_e",
    "periodo_de",
    "descrever_periodos",
    "esquivel_e",
    "esquivel_degenerou",
]


def _log1p_exp(E: np.ndarray) -> np.ndarray:
    """``ln(1 + exp(E))`` estável, vetorizado.

    Para ``E > 0`` usa ``E + log1p(exp(-E))``, evitando o overflow de
    ``exp(E)``; para ``E <= 0`` usa ``log1p(exp(E))``, evitando o cancelamento
    de ``1 + e^E`` quando ``e^E`` é minúsculo.
    """
    E = np.asarray(E, dtype=float)
    saida = np.empty_like(E)
    pos = E > 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        saida[pos] = E[pos] + np.log1p(np.exp(-E[pos]))
        saida[~pos] = np.log1p(np.exp(E[~pos]))
    return saida


@dataclass(frozen=True)
class GruposSovova:
    """Grupos adimensionais e transições de período.

    Attributes
    ----------
    Z : float
        Unidades de transferência na fase fluida [-].
    W : float
        Unidades de transferência na fase sólida [-].
    A : float
        ``W * x0 / y_r`` [-]. É este o expoente que estoura na forma ingênua.
    qm : float
        Fim do período de solubilidade [kg CO2 / kg inerte].
    qn : float
        Início do período difusional puro [kg CO2 / kg inerte].
    """

    Z: float
    W: float
    A: float
    qm: float
    qn: float


def grupos_sovova(
    *,
    Kfa0_1_s: float,
    Ksa0_1_s: float,
    sa: float,
    yr_kg_kg: float,
    x0_kg_kg: float,
    N_kg: float,
    eps: float,
    rho_co2_kg_m3: float,
    rho_solido_kg_m3: float,
    Q_kg_s: float,
) -> GruposSovova:
    """Calcula Z, W, A, q_m e q_n.

    Parameters
    ----------
    Kfa0_1_s, Ksa0_1_s : float
        Coeficientes volumétricos de transferência de massa [1/s].
    sa : float
        Fração do soluto em células intactas, ``x_k/x0`` [-].
    yr_kg_kg : float
        Solubilidade [kg óleo / kg CO2].
    x0_kg_kg : float
        Teor inicial de óleo [kg óleo / kg inerte].
    N_kg : float
        Carga de sólido livre de soluto [kg].
    eps : float
        Porosidade do leito [-].
    rho_co2_kg_m3, rho_solido_kg_m3 : float
        Densidades [kg/m3].
    Q_kg_s : float
        Vazão mássica de CO2 [kg/s].
    """
    sa = float(np.clip(sa, SA_MIN, SA_MAX))
    um_menos_eps = 1.0 - eps

    Z = Kfa0_1_s * rho_co2_kg_m3 * N_kg / (um_menos_eps * rho_solido_kg_m3 * Q_kg_s)
    W = Ksa0_1_s * N_kg / (um_menos_eps * Q_kg_s)
    A = W * x0_kg_kg / yr_kg_kg

    qm = x0_kg_kg * (1.0 - sa) / (yr_kg_kg * Z)

    # q_n - q_m = (1/W) * ln[sa + (1-sa)*e^A], via logaddexp: estável para
    # A -> 0 e A -> +inf, e degenera certo em sa -> 0 e sa -> 1.
    with np.errstate(divide="ignore"):
        qn = qm + float(np.logaddexp(np.log(sa), np.log1p(-sa) + A)) / W

    return GruposSovova(Z=Z, W=W, A=A, qm=qm, qn=qn)


def sovova_e(
    q: np.ndarray,
    *,
    x0_kg_kg: float,
    sa: float,
    yr_kg_kg: float,
    Z: float,
    W: float,
    qm: float | None = None,
    qn: float | None = None,
) -> np.ndarray:
    """Rendimento ``e(q)`` [kg óleo / kg inerte].

    Parameters
    ----------
    q : np.ndarray
        Massa específica de solvente [kg CO2 / kg inerte].
    qm, qn : float, optional
        Transições; recalculadas se omitidas.
    """
    q = np.asarray(q, dtype=float)
    sa = float(np.clip(sa, SA_MIN, SA_MAX))
    A = W * x0_kg_kg / yr_kg_kg

    if qm is None or qn is None:
        qm = x0_kg_kg * (1.0 - sa) / (yr_kg_kg * Z)
        with np.errstate(divide="ignore"):
            qn = qm + float(np.logaddexp(np.log(sa), np.log1p(-sa) + A)) / W

    e = np.empty_like(q)
    m1 = q < qm
    m2 = (q >= qm) & (q < qn)
    m3 = q >= qn

    # --- Período I: CO2 sai saturado, reta em q ---------------------------
    # 1 - exp(-Z) via -expm1(-Z) preserva precisão para Z pequeno.
    if np.any(m1):
        e[m1] = q[m1] * yr_kg_kg * (-np.expm1(-Z))

    # --- Período II: transição -------------------------------------------
    # zk = (Z/A) * [B + ln((1-sa) + sa*(-expm1(-B))) - ln(1-sa)], B = W(q-qm)
    #
    # A forma ((1-sa) + sa*(-expm1(-B))) é soma de dois termos não-negativos:
    # não há cancelamento catastrófico quando B -> 0 com sa -> 1, ao contrário
    # da forma direta (1 - sa*exp(-B)).
    if np.any(m2):
        B = W * (q[m2] - qm)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_raz = B + np.log((1.0 - sa) + sa * (-np.expm1(-B))) - np.log1p(-sa)
        zk = (Z / A) * ln_raz if A > 0.0 else np.zeros_like(B)
        e[m2] = yr_kg_kg * (q[m2] - qm * np.exp(np.minimum(zk - Z, 0.0)))

    # --- Período III: difusão interna pura --------------------------------
    # E = ln(e^A - 1) + ln(sa) + W(qm - q), com
    #     ln(e^A - 1) = A + ln(-expm1(-A))  [estável nos dois extremos de A]
    if np.any(m3):
        with np.errstate(divide="ignore", invalid="ignore"):
            E = A + np.log(-np.expm1(-A)) + np.log(sa) + W * (qm - q[m3])
        e[m3] = x0_kg_kg - (yr_kg_kg / W) * _log1p_exp(E)

    return e


def periodo_de(q: np.ndarray, qm: float, qn: float) -> np.ndarray:
    """Período de Sovová (1, 2 ou 3) de cada ``q``."""
    q = np.asarray(q, dtype=float)
    return np.where(q < qm, 1, np.where(q < qn, 2, 3))


def descrever_periodos(q: np.ndarray, qm: float, qn: float) -> str:
    """Frase pronta sobre em que período os dados caem e o que isso implica."""
    per = sorted(set(periodo_de(q, qm, qn).tolist()))
    if per == [1]:
        return ("todos no periodo I (solubilidade): e e reta em q, so o produto "
                "y_r*(1-e^-Z) e determinado; Ksa0 e sa nao entram na predicao")
    if per == [3]:
        return ("todos no periodo III (difusao interna): o periodo de "
                "solubilidade acabou antes da 1a coleta, Kfa0 nao afeta a curva")
    nomes = {1: "I", 2: "II", 3: "III"}
    return "periodos " + ", ".join(nomes[p] for p in per)


# =============================================================================
# Modelo de Esquível et al. (1999)
# =============================================================================


def esquivel_e(t_s: np.ndarray, *, e_lim_kg_kg: float, b_s: float) -> np.ndarray:
    """Rendimento ``e(t) = e_lim * t / (b + t)`` [kg óleo / kg inerte].

    Parameters
    ----------
    t_s : np.ndarray
        Tempo [s].
    e_lim_kg_kg : float
        Rendimento assintótico [kg óleo / kg inerte]. Tem teto físico: não pode
        passar de ``x0``, o teor que existe na matriz.
    b_s : float
        Constante de tempo [s]; em ``t = b`` sai metade de ``e_lim``.

    Notes
    -----
    Quando ``b`` é muito maior que o tempo do ensaio, o modelo degenera em
    reta — ``e ~ (e_lim/b) * t`` — e apenas a razão ``e_lim/b`` é determinável.
    Nesse regime a correlação entre os dois parâmetros vai a 1,0000, e reportar
    ``e_lim`` e ``b`` separadamente sugere uma precisão que os dados não têm.
    """
    t_s = np.asarray(t_s, dtype=float)
    return e_lim_kg_kg * t_s / (b_s + t_s)


def esquivel_degenerou(b_s: float, t_max_s: float) -> bool:
    """``True`` se ``b`` excede a janela experimental."""
    return b_s > t_max_s
